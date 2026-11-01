"""CONFLICTS: the model proposes, code decides, and the report states both sides."""

from __future__ import annotations

from typing import Any, cast

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LocalLLM
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.pipeline.conflicts import (
    Conflict,
    ConflictSide,
    comparable_claims,
    conflict_notes,
    disagreement_markdown,
    find_conflicts,
)
from quarry_ldr.pipeline.plan import ResearchPlan, SubQuestion
from quarry_ldr.pipeline.triage import TriagedChunk, TriageVerdict
from quarry_ldr.report.citations import CitationIndex


def _evidence(position: int, site: str, claim: str, sq: str = "sq01") -> TriagedChunk:
    return TriagedChunk(
        sub_question_id=sq,
        chunk=Chunk(
            chunk_id=f"c{position}",
            url=f"https://{site}/doc",
            text=claim,
            token_count=10,
            position=position,
            start_char=0,
            end_char=len(claim),
        ),
        verdict=TriageVerdict(relevant=True, claim=claim, evidence_span="s", confidence=0.9),
        rerank_score=1.0,
    )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        topic="t",
        sub_questions=[
            SubQuestion(
                id=f"sq{i:02d}",
                question=f"Question {i}?",
                queries=["a", "b"],
                success_criterion="c",
            )
            for i in range(1, 9)
        ],
    )


class FakeLLM:
    def __init__(self, conflicts: list[dict[str, Any]] | None = None, fail: bool = False) -> None:
        self.conflicts = conflicts or []
        self.fail = fail
        self.calls = 0

    async def complete_typed(
        self, prompt: str, schema: Any, max_tokens: int = 512, max_retries: int = 2
    ) -> Any:
        self.calls += 1
        if self.fail:
            raise ValueError("malformed")
        return schema.model_validate({"conflicts": self.conflicts})


GROWTH_A = "The market will grow at a CAGR of 46.25% through 2035."
GROWTH_B = "Deployments will grow at a CAGR of 58% over five years."


def _proposal(**overrides: Any) -> dict[str, Any]:
    return {
        "measure": "Growth rate",
        "claim_a": 1,
        "value_a": "46.25%",
        "claim_b": 2,
        "value_b": "58%",
    } | overrides


def _growth_evidence() -> list[TriagedChunk]:
    return [_evidence(1, "a.example", GROWTH_A), _evidence(2, "b.example", GROWTH_B)]


async def test_a_real_conflict_is_accepted(cfg: QuarryConfig) -> None:
    llm = FakeLLM([_proposal()])
    conflicts = await find_conflicts(_plan(), _growth_evidence(), cast(LocalLLM, llm), cfg)
    assert len(conflicts) == 1
    assert conflicts[0].measure == "Growth rate"
    assert conflicts[0].a.chunk_id == "c1"
    assert conflicts[0].b.value == "58%"


async def test_hallucinated_value_is_rejected(cfg: QuarryConfig) -> None:
    llm = FakeLLM([_proposal(value_b="99%")])
    assert await find_conflicts(_plan(), _growth_evidence(), cast(LocalLLM, llm), cfg) == []


async def test_out_of_range_claim_number_is_rejected(cfg: QuarryConfig) -> None:
    llm = FakeLLM([_proposal(claim_b=9), _proposal(claim_a=0)])
    assert await find_conflicts(_plan(), _growth_evidence(), cast(LocalLLM, llm), cfg) == []


async def test_same_site_is_not_a_conflict(cfg: QuarryConfig) -> None:
    evidence = [
        *_growth_evidence(),
        _evidence(3, "a.example", "Growth of 70% is also claimed here."),
    ]
    llm = FakeLLM([_proposal(claim_b=3, value_b="70%")])
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


async def test_rounding_and_overlapping_ranges_are_not_conflicts(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "Efficiency is 42% in tests."),
        _evidence(2, "b.example", "Efficiency is 43% in tests."),
        _evidence(3, "c.example", "Efficiency is 40-45% in practice."),
    ]
    llm = FakeLLM(
        [
            _proposal(value_a="42%", value_b="43%"),
            _proposal(claim_a=1, claim_b=3, value_a="42%", value_b="40-45%"),
        ]
    )
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


async def test_disjoint_ranges_are_a_conflict(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "Practical density is 300-350 Wh/kg."),
        _evidence(2, "b.example", "Density is above 1,000 Wh/kg."),
    ]
    llm = FakeLLM([_proposal(value_a="300-350 Wh/kg", value_b="above 1,000 Wh/kg")])
    assert len(await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg)) == 1


async def test_different_units_and_bare_years_are_rejected(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "It cost $40 per kWh in 2020."),
        _evidence(2, "b.example", "It reached 90% in 2030."),
    ]
    llm = FakeLLM(
        [
            _proposal(value_a="$40 per kWh", value_b="90%"),
            _proposal(value_a="2020", value_b="2030"),
        ]
    )
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


async def test_no_figures_means_no_model_call(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "It is cheap."),
        _evidence(2, "b.example", "It is expensive."),
    ]
    llm = FakeLLM([_proposal()])
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []
    assert llm.calls == 0


async def test_malformed_model_reply_finds_nothing_without_crashing(cfg: QuarryConfig) -> None:
    llm = FakeLLM(fail=True)
    assert await find_conflicts(_plan(), _growth_evidence(), cast(LocalLLM, llm), cfg) == []
    assert llm.calls == 1


async def test_duplicate_conflict_across_sub_questions_counts_once(cfg: QuarryConfig) -> None:
    evidence = [
        *_growth_evidence(),
        _evidence(1, "a.example", GROWTH_A, sq="sq02"),
        _evidence(2, "b.example", GROWTH_B, sq="sq02"),
    ]
    llm = FakeLLM([_proposal()])
    assert len(await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg)) == 1


def test_comparable_claims_keeps_one_figure_claim_per_chunk() -> None:
    items = [
        _evidence(1, "a.example", "Rate of 5%."),
        _evidence(1, "a.example", "Rate of 5%.", sq="sq02"),
        _evidence(2, "b.example", "No figures here."),
    ]
    assert [item.chunk.chunk_id for item in comparable_claims(items)] == ["c1"]


def _numbered() -> tuple[Conflict, CitationIndex]:
    citations = CitationIndex()
    for item in _growth_evidence():
        citations.add(item.chunk)
    conflict = Conflict(
        sub_question_id="sq01",
        measure="Growth rate",
        a=ConflictSide(chunk_id="c1", claim=GROWTH_A, value="46.25%"),
        b=ConflictSide(chunk_id="c2", claim=GROWTH_B, value="58%"),
    )
    return conflict, citations


def test_notes_are_scoped_and_empty_when_nothing_applies() -> None:
    conflict, citations = _numbered()
    assert "[1] reports 46.25%" in conflict_notes([conflict], citations)
    assert conflict_notes([conflict], citations, ["sq05"]) == ""
    assert conflict_notes([], citations) == ""


def test_disagreement_markdown_states_both_sides_with_citations() -> None:
    conflict, citations = _numbered()
    line = disagreement_markdown([conflict], citations)
    assert line.startswith("- **Growth rate:**")
    assert "[1]" in line and "[2]" in line
    citations.validate_markdown(line)


def test_unnumbered_chunks_are_left_out() -> None:
    conflict, _ = _numbered()
    assert disagreement_markdown([conflict], CitationIndex()) == ""
    assert conflict_notes([conflict], CitationIndex()) == ""
