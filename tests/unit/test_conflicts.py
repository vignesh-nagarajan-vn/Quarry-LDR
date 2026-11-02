"""CONFLICTS: the model proposes, code decides, and the report states both sides."""

from __future__ import annotations

from typing import Any, cast

import pytest

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LocalLLM
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.pipeline.conflicts import (
    Conflict,
    ConflictSide,
    _unit,
    cite_conflict_sides,
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
    def __init__(
        self, conflicts: list[dict[str, Any]] | None = None, fail: bool = False, same: bool = False
    ) -> None:
        self.conflicts = conflicts or []
        self.fail = fail
        self.same = same
        self.calls = 0

    async def complete_typed(
        self, prompt: str, schema: Any, max_tokens: int = 512, max_retries: int = 2
    ) -> Any:
        self.calls += 1
        if self.fail:
            raise ValueError("malformed")
        return schema.model_validate({"conflicts": self.conflicts, "same": self.same})


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
    assert llm.calls == 2  # the coded pair's confirmation, then the listing


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


async def test_a_bound_does_not_disagree_with_a_value_inside_it(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "Round-trip efficiency of approximately 40% is typical."),
        _evidence(2, "b.example", "Conventional cells have an efficiency of less than 50 percent."),
    ]
    llm = FakeLLM(
        [_proposal(value_a="approximately 40%", value_b="less than 50 percent")], same=True
    )
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


async def test_claims_sharing_a_figure_do_not_disagree(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(
            1,
            "a.example",
            "Drugs can lead to weight loss of over 20% in some studies, with participants "
            "losing an average of 10% to 15% over a year.",
        ),
        _evidence(2, "b.example", "Patients reached a peak weight loss of 12% at 18 months."),
    ]
    llm = FakeLLM([_proposal(value_a="over 20%", value_b="12%")], same=True)
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


@pytest.mark.parametrize(
    ("written", "plain"),
    [("50% to 60%", "%"), ("10 to 20 Wh/kg", "wh/kg"), ("5 hours to 8 hours", "hours")],
)
def test_a_range_written_out_keeps_its_unit(written: str, plain: str) -> None:
    assert _unit(written) == plain


async def test_other_figures_that_differ_keep_a_conflict(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "Efficiency of about 40%, against 90% for lithium-ion."),
        _evidence(2, "b.example", "Efficiency of 50% to 60%, with 5% lost to self-discharge."),
    ]
    llm = FakeLLM([_proposal(value_a="about 40%", value_b="50% to 60%")], same=True)
    assert len(await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg)) == 1


async def test_a_goal_does_not_contradict_a_measured_figure(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "Cells reach an energy density of 75 Wh/kg."),
        _evidence(2, "b.example", "A new cell being developed has an energy density of 250 Wh/kg."),
    ]
    llm = FakeLLM([_proposal(value_a="75 Wh/kg", value_b="250 Wh/kg")], same=True)
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


def _efficiency_evidence() -> list[TriagedChunk]:
    return [
        _evidence(1, "a.example", "These cells have a round-trip efficiency of about 40%.", "sq01"),
        _evidence(2, "b.example", "These cells have a round-trip efficiency of 50-60%.", "sq05"),
    ]


async def test_code_pairs_a_measure_across_sub_questions(cfg: QuarryConfig) -> None:
    llm = FakeLLM(same=True)
    conflicts = await find_conflicts(_plan(), _efficiency_evidence(), cast(LocalLLM, llm), cfg)
    assert len(conflicts) == 1
    assert conflicts[0].measure == "round-trip efficiency"
    assert (conflicts[0].a.value, conflicts[0].b.value) == ("about 40%", "50-60%")


async def test_a_range_written_with_two_units_is_read_whole(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "These cells have a round-trip efficiency of 50-60%."),
        _evidence(2, "b.example", "These cells have a round-trip efficiency of 40% to 50%."),
    ]
    llm = FakeLLM(same=True)
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


async def test_code_pairs_figures_with_word_units(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "These cells have a cycle life of 2000 cycles.", "sq01"),
        _evidence(2, "b.example", "These cells have a cycle life of 10,000 cycles.", "sq02"),
        _evidence(3, "c.example", "These cells have an efficiency of 40 percent.", "sq03"),
        _evidence(4, "d.example", "These cells have an efficiency of 60 percent.", "sq04"),
    ]
    llm = FakeLLM(same=True)
    conflicts = await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg)
    assert {(c.measure, c.a.value, c.b.value) for c in conflicts} == {
        ("cycle life", "2000 cycles", "10,000 cycles"),
        ("efficiency", "40 percent", "60 percent"),
    }


async def test_a_coded_pair_needs_the_model_to_confirm_it(cfg: QuarryConfig) -> None:
    llm = FakeLLM(same=False)
    assert await find_conflicts(_plan(), _efficiency_evidence(), cast(LocalLLM, llm), cfg) == []


async def test_a_malformed_confirmation_keeps_no_conflict(cfg: QuarryConfig) -> None:
    llm = FakeLLM(fail=True)
    assert await find_conflicts(_plan(), _efficiency_evidence(), cast(LocalLLM, llm), cfg) == []


async def test_coded_pairs_stay_within_the_per_measure_cap(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(n, f"s{n}.example", f"Cells have a round-trip efficiency of {n * 20}%.")
        for n in range(1, 5)
    ]
    llm = FakeLLM(same=True)
    conflicts = await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg)
    assert len(conflicts) == 2


async def test_two_sites_are_needed_for_a_coded_pair(cfg: QuarryConfig) -> None:
    evidence = [
        _evidence(1, "a.example", "Cells have a round-trip efficiency of 40%."),
        _evidence(2, "a.example", "Cells have a round-trip efficiency of 60%."),
    ]
    llm = FakeLLM(same=True)
    assert await find_conflicts(_plan(), evidence, cast(LocalLLM, llm), cfg) == []


def test_sides_outside_the_corpus_are_numbered_after_it() -> None:
    evidence = _efficiency_evidence()
    citations = CitationIndex()
    citations.add(evidence[0].chunk)
    conflict = Conflict(
        sub_question_id="sq01",
        measure="round-trip efficiency",
        a=ConflictSide(chunk_id="c1", claim="a", value="about 40%"),
        b=ConflictSide(chunk_id="c2", claim="b", value="50-60%"),
    )
    cite_conflict_sides([conflict], evidence, citations)
    assert citations.number_for("c1") == 1
    assert citations.number_for("c2") == 2
    cite_conflict_sides([conflict], evidence, citations)
    assert len(citations) == 2
