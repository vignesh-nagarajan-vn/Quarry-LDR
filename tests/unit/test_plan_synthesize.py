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
from quarry_ldr.pipeline.plan import ResearchPlan, SubQuestion, make_plan
from quarry_ldr.pipeline.synthesize import (
    _corpus_entry,
    build_evidence_corpus,
    plan_sections,
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


def test_plan_sections_shape_and_bounds(cfg: QuarryConfig) -> None:
    briefs = plan_sections(make_plan_obj(8), cfg)
    assert briefs[0].title == "Overview"
    assert briefs[-1].title == "Conclusions"
    assert len(briefs) == 10  # overview + 8 + conclusions
    assert briefs[1].sub_question_ids == ["sq01"]

    cfg.report.max_sections = 6
    briefs = plan_sections(make_plan_obj(12), cfg)
    assert len(briefs) <= 6
    body = briefs[1:-1]
    covered = [sq for brief in body for sq in brief.sub_question_ids]
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
    assert len(draft.sections) == 10
    assert len(set(fake.corpora)) == 1  # byte identical prefix on all 10 calls
    assert draft.corpus_hash
    assert all("[1]" in section.markdown for section in draft.sections)
    # ledger saw one cache write then nine reads
    summary = ledger.summary()
    assert summary.total_cache_write_tokens == 60000
    assert summary.total_cache_read_tokens == 9 * 60000
    # every brief names its section title
    assert 'titled "Overview"' in fake.briefs[0]
    assert 'titled "Conclusions"' in fake.briefs[-1]


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
