"""synthesize_local: sliced corpora, global citation numbers, marker hygiene."""

from __future__ import annotations

from typing import Any, cast

import pytest

from quarry_ldr.config import QuarryConfig, SourceSettings
from quarry_ldr.ingest.chunk import Chunk, HeuristicTokenCounter, make_chunk_id
from quarry_ldr.pipeline.plan import ResearchPlan, SubQuestion
from quarry_ldr.pipeline.synthesize import (
    DraftReport,
    ReportSection,
    _section_corpus,
    _strip_invalid_citations,
    _strip_leading_heading,
    build_evidence_corpus,
    edit_sections_local,
    synthesize_local,
)
from quarry_ldr.pipeline.triage import TriagedChunk, TriageVerdict
from quarry_ldr.providers.base import Provider
from quarry_ldr.report.citations import CitationIndex


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


def make_evidence(n: int = 3, text_len: int = 24) -> list[TriagedChunk]:
    return [
        TriagedChunk(
            sub_question_id=f"sq{(i % 2) + 1:02d}",
            chunk=make_chunk("x" * text_len + f" passage {i}", position=i),
            verdict=TriageVerdict(
                relevant=True, claim=f"Claim {i}.", evidence_span=f"span {i}", confidence=0.9
            ),
            rerank_score=float(10 - i),
        )
        for i in range(n)
    ]


def group_by_sq(evidence: list[TriagedChunk]) -> dict[str, list[TriagedChunk]]:
    grouped: dict[str, list[TriagedChunk]] = {}
    ordered = sorted(evidence, key=lambda t: (t.sub_question_id, -t.rerank_score, t.chunk.chunk_id))
    for item in ordered:
        grouped.setdefault(item.sub_question_id, []).append(item)
    return grouped


class FakeSectionProvider:
    """Provider-shaped fake for the local path: records every prompt and
    max_tokens, returns scripted section markdown through the schema."""

    def __init__(self, scripted: list[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.max_tokens_seen: list[int] = []
        self._scripted = scripted
        self.calls = 0

    async def complete_typed(self, **kwargs: Any) -> Any:
        self.prompts.append(kwargs["prompt"])
        self.max_tokens_seen.append(kwargs["max_tokens"])
        if self._scripted is None:
            text = "Section cites [1]."
        else:
            text = self._scripted[min(self.calls, len(self._scripted) - 1)]
        self.calls += 1
        return kwargs["schema"].model_validate({"markdown": text})


async def test_local_sections_use_sliced_corpora_only(cfg: QuarryConfig) -> None:
    fake = FakeSectionProvider()
    citations = CitationIndex()
    draft = await synthesize_local(
        make_plan_obj(8), make_evidence(4), cast(Provider, fake), cfg, citations
    )
    assert len(draft.sections) == 8  # one per sub-question, no overview or conclusions
    assert draft.corpus_hash == ""
    assert len(citations) == 4  # numbered up front from the full selection
    assert all(prompt.startswith("EVIDENCE CORPUS") for prompt in fake.prompts)
    assert all("Reply with JSON only" in prompt for prompt in fake.prompts)
    assert set(fake.max_tokens_seen) == {cfg.synth.section_max_tokens}


def test_section_corpus_numbers_come_from_global_index() -> None:
    evidence = make_evidence(4)
    citations = CitationIndex()
    global_corpus = build_evidence_corpus(evidence, citations)
    grouped = group_by_sq(evidence)
    section = _section_corpus(
        grouped, ["sq02"], citations, budget_tokens=100_000, source=SourceSettings()
    )
    for item in grouped["sq02"]:
        number = citations.add(item.chunk)  # returns the existing number
        assert f"[{number}] source:" in section
        assert f"[{number}] source:" in global_corpus
    assert "Sub-question sq01" not in section


async def test_section_prompts_respect_budget(cfg: QuarryConfig) -> None:
    cfg.synth.section_budget_tokens = 120
    counter = HeuristicTokenCounter()
    fake = FakeSectionProvider()
    # Each entry is ~115 heuristic tokens with overhead, so roughly one fits
    # per section; twelve exist across the two sub-questions.
    evidence = make_evidence(12, text_len=300)
    await synthesize_local(make_plan_obj(8), evidence, cast(Provider, fake), cfg, CitationIndex())
    scaffold_allowance = 320  # SECTION_PROMPT (with specifics and length rules) + local rules
    for prompt in fake.prompts:
        if prompt.startswith("EVIDENCE CORPUS"):
            assert counter.count(prompt) <= cfg.synth.section_budget_tokens + scaffold_allowance


def test_strip_leading_heading_drops_model_emitted_titles() -> None:
    """claude-opus-5 was observed opening sections with # or ## titles the
    renderer already injects; ###+ subsections are content and must pass."""
    assert _strip_leading_heading("# Overview\n\nBody [1].") == "Body [1]."
    assert _strip_leading_heading("## A long question title\nBody.") == "Body."
    assert _strip_leading_heading("### Subsection\n\nBody.").startswith("### Subsection")
    assert _strip_leading_heading("Plain body [2].") == "Plain body [2]."


def test_strip_invalid_citations_removes_unknown_markers() -> None:
    citations = CitationIndex()
    build_evidence_corpus(make_evidence(2), citations)
    text = "Known [1] and unknown [99] and known [2]."
    cleaned = _strip_invalid_citations(text, citations)
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert "[99]" not in cleaned
    citations.validate_markdown(cleaned)  # must not raise


async def test_invented_markers_never_reach_the_draft(cfg: QuarryConfig) -> None:
    fake = FakeSectionProvider(scripted=["Cites [1] and [99]."])
    citations = CitationIndex()
    draft = await synthesize_local(
        make_plan_obj(8), make_evidence(2), cast(Provider, fake), cfg, citations
    )
    assert all("[99]" not in section.markdown for section in draft.sections)
    assert all("[1]" in section.markdown for section in draft.sections)
    for section in draft.sections:
        citations.validate_markdown(section.markdown)


async def test_empty_section_retries_once(cfg: QuarryConfig) -> None:
    fake = FakeSectionProvider(scripted=["", "Recovered [1].", "ok [1]."])
    draft = await synthesize_local(
        make_plan_obj(8), make_evidence(2), cast(Provider, fake), cfg, CitationIndex()
    )
    assert draft.sections[0].markdown == "Recovered [1]."
    assert fake.calls == 9  # 8 sections + 1 retry


def _two_sections() -> DraftReport:
    return DraftReport(
        topic="t",
        sections=[
            ReportSection(title="Plants", markdown="Halvard Grid runs a 5 MW plant in Tromso [1]."),
            ReportSection(title="Costs", markdown="Costs reached 20 dollars per kWh in 2027 [2]."),
        ],
    )


async def test_local_edit_applies_when_markers_and_length_hold(cfg: QuarryConfig) -> None:
    fake = FakeSectionProvider(
        scripted=[
            "Halvard Grid operates a 5 MW plant in Tromso [1].",
            "By 2027 costs had reached 20 dollars per kWh [2].",
        ]
    )
    edited = await edit_sections_local(_two_sections(), cast(Provider, fake), cfg)
    assert [s.markdown for s in edited.sections] == fake._scripted
    assert 'titled "Plants"' in fake.prompts[0]
    assert "Halvard Grid runs a 5 MW plant in Tromso [1]." in fake.prompts[0]


@pytest.mark.parametrize(
    "reply",
    [
        "Halvard Grid operates a 5 MW plant in Tromso.",  # marker dropped
        "Halvard Grid operates a 5 MW plant in Tromso [1][3].",  # marker added
        "A plant [1].",  # shrunk to a rewrite
        "Halvard Grid operates a 5 MW plant in Tromso, Norway, the largest such plant "
        "anywhere in the region, built after years of planning and debate [1].",  # grew
    ],
)
async def test_local_edit_is_discarded_when_a_guard_fails(cfg: QuarryConfig, reply: str) -> None:
    fake = FakeSectionProvider(scripted=[reply, "Costs reached 20 dollars per kWh by 2027 [2]."])
    edited = await edit_sections_local(_two_sections(), cast(Provider, fake), cfg)
    assert edited.sections[0] == _two_sections().sections[0]
    assert edited.sections[1].markdown == "Costs reached 20 dollars per kWh by 2027 [2]."


async def test_local_edit_failure_keeps_the_draft(cfg: QuarryConfig) -> None:
    class Broken:
        async def complete_typed(self, **kwargs: Any) -> Any:
            raise ValueError("malformed JSON")

    edited = await edit_sections_local(_two_sections(), cast(Provider, Broken()), cfg)
    assert edited == _two_sections()
