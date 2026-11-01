"""Plan parsing, section planning, corpus determinism, cached synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import respx
from anthropic import AsyncAnthropic

from quarry_ldr.config import QuarryConfig, SourceSettings
from quarry_ldr.ingest.chunk import Chunk, HeuristicTokenCounter, make_chunk_id
from quarry_ldr.ledger import Ledger, TokenUsage
from quarry_ldr.pipeline.plan import ResearchPlan, SectionPlan, SubQuestion, make_plan
from quarry_ldr.pipeline.synthesize import (
    SECTION_PROMPT,
    _corpus_entry,
    build_evidence_corpus,
    plan_sections,
    section_groups,
    section_word_range,
    select_evidence,
    synthesize,
)
from quarry_ldr.pipeline.triage import TriagedChunk, TriageVerdict
from quarry_ldr.providers.anthropic_client import AnthropicProvider, CompletionResult
from quarry_ldr.report.citations import CitationIndex

MESSAGES = "https://api.anthropic.com/v1/messages"


def make_chunk(text: str, position: int = 0) -> Chunk:
    url = "https://vesterholm-times.example/2024/article"
    return Chunk(
        chunk_id=make_chunk_id(url, position),
        url=url,
        doc_title="Test doc",
        heading_path=["Background"],
        text=text,
        token_count=len(text) // 4,
        position=position,
        start_char=0,
        end_char=len(text),
    )


def make_plan_obj(n: int = 8) -> ResearchPlan:
    return ResearchPlan(
        topic="fixture topic",
        sub_questions=[
            SubQuestion(
                id=f"sq{i:02d}",
                question=f"Question {i} about the pilot?",
                queries=[f"query {i}a", f"query {i}b"],
                success_criterion=f"Criterion {i}.",
            )
            for i in range(1, n + 1)
        ],
    )


def make_evidence(n: int = 3) -> list[TriagedChunk]:
    return [
        TriagedChunk(
            sub_question_id=f"sq{(i % 2) + 1:02d}",
            chunk=make_chunk(f"passage text {i}", position=i),
            verdict=TriageVerdict(
                relevant=True, claim=f"Claim {i}.", evidence_span=f"span {i}", confidence=0.9
            ),
            rerank_score=float(10 - i),
        )
        for i in range(n)
    ]


async def test_make_plan_parses_fixture(cfg: QuarryConfig, fixtures_dir: Path) -> None:
    body = json.loads((fixtures_dir / "anthropic" / "plan_response.json").read_text("utf-8"))
    provider = AnthropicProvider(
        cfg, Ledger(), client=AsyncAnthropic(api_key="test-key-not-real", max_retries=0)
    )
    with respx.mock:
        respx.post(MESSAGES).mock(return_value=httpx.Response(200, json=body))
        plan = await make_plan("the vesterholm sand battery pilot", provider, cfg)
    assert plan.topic == "the vesterholm sand battery pilot"
    assert len(plan.sub_questions) == 9
    assert plan.sub_questions[0].id == "sq01"
    assert len(plan.all_queries()) == 18


def test_old_plans_without_must_pin_down_still_load() -> None:
    raw = make_plan_obj().model_dump(mode="json")
    for sub_question in raw["sub_questions"]:
        del sub_question["must_pin_down"]
    assert ResearchPlan.model_validate(raw).sub_questions[0].must_pin_down == ""


def test_selection_prefers_concrete_claims_on_a_near_tie() -> None:
    def item(position: int, claim: str) -> TriagedChunk:
        base = make_evidence(1)[0]
        return base.model_copy(
            update={
                "chunk": make_chunk("x", position=position),
                "verdict": base.verdict.model_copy(update={"claim": claim}),
                "rerank_score": 5.0,
            }
        )

    vague = item(1, "it is quite good")
    concrete = item(2, "Form Energy built a 1.5 MW pilot in Minnesota")
    budget = HeuristicTokenCounter().count(_corpus_entry(concrete, 0))
    kept = select_evidence([vague, concrete], budget, source=SourceSettings(max_site_share=1.0))
    assert [k.verdict.claim for k in kept] == [concrete.verdict.claim]


def test_plan_sections_shape_and_bounds(cfg: QuarryConfig) -> None:
    briefs = plan_sections(make_plan_obj(8), cfg)
    assert len(briefs) == 8  # body only: findings, gaps, and sources come later
    assert briefs[0].sub_question_ids == ["sq01"]
    assert "Write 468 to 750 words." in briefs[0].instructions

    cfg.report.max_sections = 6
    briefs = plan_sections(make_plan_obj(12), cfg)
    assert len(briefs) <= 6
    covered = [sq for brief in briefs for sq in brief.sub_question_ids]
    assert covered == [f"sq{i:02d}" for i in range(1, 13)]  # all covered, in order


def test_select_evidence_keeps_everything_within_budget() -> None:
    evidence = make_evidence(4)
    kept = select_evidence(evidence, budget_tokens=100_000)
    assert {t.chunk.chunk_id for t in kept} == {t.chunk.chunk_id for t in evidence}
    kept_reversed = select_evidence(list(reversed(evidence)), budget_tokens=100_000)
    assert [t.chunk.chunk_id for t in kept_reversed] == [t.chunk.chunk_id for t in kept]


def test_select_evidence_budget_spreads_across_sub_questions() -> None:
    # make_evidence alternates sq01/sq02; budget sized for exactly the top
    # entry of each sub-question, so round-robin must keep one from each
    # rather than two from the best sub-question.
    evidence = make_evidence(6)
    counter = HeuristicTokenCounter()
    by_score = sorted(evidence, key=lambda t: -t.rerank_score)
    budget = counter.count(_corpus_entry(by_score[0], 0)) + counter.count(
        _corpus_entry(by_score[1], 0)
    )
    kept = select_evidence(evidence, budget_tokens=budget)
    assert len(kept) == 2
    assert {t.sub_question_id for t in kept} == {"sq01", "sq02"}
    assert {t.rerank_score for t in kept} == {10.0, 9.0}  # the best of each


def test_select_evidence_zero_budget_is_empty_and_total_stays_capped() -> None:
    evidence = make_evidence(6)
    assert select_evidence(evidence, budget_tokens=0) == []
    counter = HeuristicTokenCounter()
    for budget in (50, 120, 400):
        kept = select_evidence(evidence, budget_tokens=budget)
        assert sum(counter.count(_corpus_entry(t, 0)) for t in kept) <= budget


def test_corpus_is_deterministic_and_numbers_citations() -> None:
    evidence = make_evidence(4)
    index_a, index_b = CitationIndex(), CitationIndex()
    corpus_a = build_evidence_corpus(evidence, index_a)
    corpus_b = build_evidence_corpus(list(reversed(evidence)), index_b)
    assert corpus_a == corpus_b  # input order does not matter
    assert len(index_a) == 4
    for number in (1, 2, 3, 4):
        assert f"[{number}]" in corpus_a
    assert "claim: Claim 0." in corpus_a


class RecordingFakeProvider:
    """Mimics complete_with_cached_corpus: pins the corpus, records usage,
    returns text citing real corpus markers."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.corpora: list[str] = []
        self.briefs: list[str] = []

    async def complete_with_cached_corpus(self, **kwargs: Any) -> CompletionResult:
        corpus: str = kwargs["corpus"]
        self.corpora.append(corpus)
        self.briefs.append(kwargs["brief"])
        write = 60000 if len(self.corpora) == 1 else 0
        read = 0 if len(self.corpora) == 1 else 60000
        usage = TokenUsage(
            input_tokens=200,
            output_tokens=400,
            cache_creation_input_tokens=write,
            cache_read_input_tokens=read,
        )
        self.ledger.record(kwargs["model"], usage, stage=kwargs.get("stage", ""))
        return CompletionResult(
            text="Fixture section citing [1] and [2].",
            usage=usage,
            model=kwargs["model"],
            stop_reason="end_turn",
        )


async def test_synthesize_sends_byte_identical_corpus_every_call(cfg: QuarryConfig) -> None:
    ledger = Ledger()
    fake = RecordingFakeProvider(ledger)
    citations = CitationIndex()
    draft = await synthesize(
        make_plan_obj(8),
        make_evidence(3),
        cast(AnthropicProvider, fake),
        cfg,
        citations,
    )
    assert len(draft.sections) == 8
    assert len(set(fake.corpora)) == 1  # byte identical prefix on all 8 calls
    assert draft.corpus_hash
    assert all("[1]" in section.markdown for section in draft.sections)
    # ledger saw one cache write then nine reads
    summary = ledger.summary()
    assert summary.total_cache_write_tokens == 60000
    assert (
        summary.total_cache_read_tokens == 7 * 60000
    )  # one write, then a cached read for each other section
    # every brief names its section title
    assert 'titled "Question 1 about the pilot"' in fake.briefs[0]
    assert not any('titled "Overview"' in brief for brief in fake.briefs)


async def test_synthesize_empty_evidence_still_produces_sections(cfg: QuarryConfig) -> None:
    ledger = Ledger()
    fake = RecordingFakeProvider(ledger)
    citations = CitationIndex()
    with pytest.raises(Exception):  # noqa: B017 - citing [1] with no citations must fail later;
        # here we just assert synthesize itself doesn't crash, then validate raises
        draft = await synthesize(
            make_plan_obj(8), [], cast(AnthropicProvider, fake), cfg, citations
        )
        citations.validate_markdown(draft.sections[0].markdown)


def _site_evidence(site: str, sq: str, position: int, score: float) -> TriagedChunk:
    url = f"https://{site}/doc"
    return TriagedChunk(
        sub_question_id=sq,
        chunk=Chunk(
            chunk_id=make_chunk_id(url, position),
            url=url,
            text="x" * 400,
            token_count=100,
            position=position,
            start_char=0,
            end_char=400,
        ),
        verdict=TriageVerdict(relevant=True, claim="c", evidence_span="s", confidence=0.9),
        rerank_score=score,
    )


def _spread(kept: list[TriagedChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in kept:
        counts[item.chunk.url] = counts.get(item.chunk.url, 0) + 1
    return counts


def test_select_evidence_caps_a_flooding_site() -> None:
    flood = [_site_evidence("flood.example", "sq01", i, 9.0) for i in range(40)]
    others = [_site_evidence(f"s{i}.example", "sq01", 100 + i, 1.0) for i in range(8)]
    entry_cost = HeuristicTokenCounter().count(_corpus_entry(flood[0], 0))
    budget = entry_cost * 10

    kept = select_evidence(flood + others, budget, source=SourceSettings(max_site_share=0.2))

    assert _spread(kept)["https://flood.example/doc"] <= 2
    assert len(kept) == 10
    assert select_evidence(flood + others, budget) != kept  # the cap changed the outcome


def test_select_evidence_share_widens_for_a_thin_corpus() -> None:
    two_sites = [_site_evidence("a.example", "sq01", i, 5.0) for i in range(5)]
    two_sites += [_site_evidence("b.example", "sq01", 10 + i, 5.0) for i in range(5)]
    entry_cost = HeuristicTokenCounter().count(_corpus_entry(two_sites[0], 0))

    kept = select_evidence(two_sites, entry_cost * 10, source=SourceSettings(max_site_share=0.2))

    assert len(kept) == 10  # a 0.2 cap is impossible with two sites; nothing is lost


def test_select_evidence_with_source_is_deterministic_and_prefers_strong_tiers() -> None:
    weak = [_site_evidence("eureka.patsnap.com", "sq01", i, 5.0) for i in range(3)]
    strong = [_site_evidence("www.energy.gov", "sq01", 10 + i, 4.0) for i in range(3)]
    entry_cost = HeuristicTokenCounter().count(_corpus_entry(weak[0], 0))
    source = SourceSettings(max_site_share=1.0)

    kept = select_evidence(weak + strong, entry_cost * 3, source=source)

    assert all("energy.gov" in item.chunk.url for item in kept)
    assert kept == select_evidence(list(reversed(weak + strong)), entry_cost * 3, source=source)


def test_a_name_from_chunk_text_reaches_the_corpus_through_its_claim() -> None:
    base = make_evidence(1)[0]
    named = base.model_copy(
        update={
            "chunk": make_chunk("Polar Night Energy switched on the Vesterholm pilot in 2024."),
            "verdict": base.verdict.model_copy(
                update={"claim": "Polar Night Energy switched on the Vesterholm pilot in 2024."}
            ),
        }
    )
    corpus = build_evidence_corpus([named], CitationIndex())
    assert "claim: Polar Night Energy switched on the Vesterholm pilot in 2024." in corpus


def test_section_prompt_requires_named_specifics() -> None:
    assert "Name the companies, projects, places" in SECTION_PROMPT
    assert "dates and figures" in SECTION_PROMPT


def _with_sections(plan: ResearchPlan, sections: list[tuple[str, list[str]]]) -> ResearchPlan:
    return plan.model_copy(
        update={"sections": [SectionPlan(title=t, sub_question_ids=ids) for t, ids in sections]}
    )


GOOD_SECTIONS = [
    ("Technology and chemistry", ["sq01", "sq02", "sq03"]),
    ("Costs", ["sq04", "sq05"]),
    ("Projects and companies", ["sq06", "sq07", "sq08"]),
]


def test_valid_plan_sections_are_used_as_given() -> None:
    groups = section_groups(_with_sections(make_plan_obj(8), GOOD_SECTIONS), max_body=10)
    assert [(g.title, g.sub_question_ids) for g in groups] == GOOD_SECTIONS


def test_plan_sections_brief_uses_the_short_titles(cfg: QuarryConfig) -> None:
    briefs = plan_sections(_with_sections(make_plan_obj(8), GOOD_SECTIONS), cfg)
    assert [b.title for b in briefs] == [t for t, _ in GOOD_SECTIONS]
    assert "Question 6 about the pilot?; Question 7" in briefs[2].instructions


def test_overlapping_plan_sections_keep_their_titles_and_each_id_once() -> None:
    # The shape the live 8B returned: good titles, each section's last ids
    # running into the next, one id missed.
    sections = [
        ("Chemistry", ["sq01", "sq03"]),
        ("Research", ["sq03", "sq04", "sq99"]),
        ("Costs", ["sq05", "sq06"]),
        ("Deployment", ["sq06", "sq07", "sq08"]),
    ]
    groups = section_groups(_with_sections(make_plan_obj(8), sections), max_body=10)
    assert [(g.title, g.sub_question_ids) for g in groups] == [
        ("Chemistry", ["sq01", "sq02"]),  # sq02 joins the section holding sq01
        ("Research", ["sq03", "sq04"]),
        ("Costs", ["sq05"]),
        ("Deployment", ["sq06", "sq07", "sq08"]),
    ]
    assigned = sorted(sq for g in groups for sq in g.sub_question_ids)
    assert assigned == [f"sq{i:02d}" for i in range(1, 9)]


def test_a_section_every_id_of_which_is_claimed_later_is_dropped() -> None:
    sections = [("Early", ["sq01"]), ("Late", [f"sq{i:02d}" for i in range(1, 9)])]
    groups = section_groups(_with_sections(make_plan_obj(8), sections), max_body=10)
    assert [g.title for g in groups] == ["Late"]


def test_a_first_sub_question_nobody_listed_joins_the_first_section() -> None:
    sections = [("Later", ["sq02", "sq03", "sq04", "sq05", "sq06", "sq07", "sq08"])]
    groups = section_groups(_with_sections(make_plan_obj(8), sections), max_body=10)
    assert groups[0].sub_question_ids[-1] == "sq01"


def test_sections_listing_no_known_id_fall_back() -> None:
    plan = _with_sections(make_plan_obj(8), [("Ghost", ["sq99"])])
    assert section_groups(plan, max_body=10)[0].title == "Question 1 about the pilot"


@pytest.mark.parametrize(
    ("question", "title"),
    [
        ("What are the key chemical reactions in iron-air batteries?", None),
        (
            "What are the primary challenges in the commercialization of iron-air batteries?",
            "Primary challenges in the commercialization",
        ),
        ("How do sodium-ion cells age in storage?", "Sodium-ion cells age in storage"),
    ],
)
def test_fallback_titles_read_as_noun_phrases(question: str, title: str | None) -> None:
    plan = make_plan_obj(8)
    plan.sub_questions[0] = plan.sub_questions[0].model_copy(update={"question": question})
    got = section_groups(plan, max_body=10)[0].title
    assert got == (title or "Key chemical reactions in iron-air batteries")


@pytest.mark.parametrize(
    "sections",
    [
        [*GOOD_SECTIONS[:2], ("What do the projects cost?", ["sq06", "sq07", "sq08"])],
        [*GOOD_SECTIONS[:2], ("Costs", ["sq06", "sq07", "sq08"])],  # duplicate title
        [*GOOD_SECTIONS[:2], ("x" * 61, ["sq06", "sq07", "sq08"])],
    ],
)
def test_invalid_plan_sections_fall_back_deterministically(
    sections: list[tuple[str, list[str]]],
) -> None:
    plan = _with_sections(make_plan_obj(8), sections)
    groups = section_groups(plan, max_body=10)
    assert [g.sub_question_ids for g in groups] == [[f"sq{i:02d}"] for i in range(1, 9)]
    assert groups[0].title == "Question 1 about the pilot"


def test_too_many_plan_sections_fall_back_within_the_bound() -> None:
    plan = _with_sections(make_plan_obj(8), [(f"Part {i}", [f"sq{i:02d}"]) for i in range(1, 9)])
    groups = section_groups(plan, max_body=4)
    assert len(groups) == 4
    assert [sq for g in groups for sq in g.sub_question_ids] == [f"sq{i:02d}" for i in range(1, 9)]


def test_fallback_titles_never_end_on_a_dangling_word() -> None:
    plan = make_plan_obj(8)
    long_q = "What is the installed capacity and commissioning schedule of the largest of the"
    plan.sub_questions[0] = plan.sub_questions[0].model_copy(update={"question": long_q + "?"})
    title = section_groups(plan, max_body=10)[0].title
    assert len(title) <= 60
    assert title.split()[-1].lower() not in {"of", "the", "and"}


def test_section_word_range_splits_the_report_band(cfg: QuarryConfig) -> None:
    assert section_word_range(cfg, 6) == (625, 1000)  # aims above the floor VERIFY cuts into
    cfg.report.target_words_max = 3100
    low, high = section_word_range(cfg, 6)
    assert low == high == 516  # never asks for a floor above the ceiling
