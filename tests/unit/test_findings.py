"""Key findings, confidence labels, and the evidence-gaps section."""

from __future__ import annotations

from typing import Any, cast

import pytest

from quarry_ldr.config import QuarryConfig, SourceSettings
from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.ledger import CostCapExceeded
from quarry_ldr.pipeline.findings import (
    GAPS_TITLE,
    KEY_FINDINGS_TITLE,
    KeyFindingsPayload,
    confidence_label,
    gaps_section,
    label_findings,
    write_key_findings,
)
from quarry_ldr.pipeline.gap import CoverageAssessment, GapAnalysis
from quarry_ldr.pipeline.plan import ResearchPlan, SubQuestion
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.providers.base import Provider
from quarry_ldr.report.citations import CitationIndex

# Citation n points at site s<n>.example, except 4 and 5, which share s4.
SITES = ["s1", "s2", "s3", "s4", "s4", "s6"]
STRONG = SourceSettings(overrides={"s1.example": "scholarly"})


def _index() -> CitationIndex:
    index = CitationIndex()
    for i, site in enumerate(SITES):
        index.add(
            Chunk(
                chunk_id=f"c{i}",
                url=f"https://{site}.example/{i}",
                text="t",
                token_count=1,
                position=i,
                start_char=0,
                end_char=1,
            )
        )
    return index


BODY = DraftReport(
    topic="iron-air batteries",
    sections=[
        ReportSection(title=f"Part {i}", markdown=f"Halvard Grid fact {i} holds [{i}]. More.")
        for i in range(1, 7)
    ],
)


class FakeWriter:
    def __init__(self, reply: KeyFindingsPayload | Exception) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def complete_typed(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


async def _write(reply: KeyFindingsPayload | Exception, cfg: QuarryConfig) -> ReportSection:
    writer = FakeWriter(reply)
    section = await write_key_findings(BODY, cast(Provider, writer), cfg, _index(), 900)
    assert writer.calls[0]["max_tokens"] == 900
    assert writer.calls[0]["stage"] == "findings"
    assert "Halvard Grid fact 3 holds [3]" in writer.calls[0]["prompt"]
    return section


async def test_cited_findings_survive_and_invented_markers_go(cfg: QuarryConfig) -> None:
    reply = KeyFindingsPayload(
        bottom_line="Iron-air is early but real [1] [99].",
        findings=[f"Finding {i} [{i}]." for i in range(1, 6)] + ["Uncited claim.", "Bad [99]."],
    )
    section = await _write(reply, cfg)
    assert section.title == KEY_FINDINGS_TITLE
    lines = section.markdown.splitlines()
    assert lines[0] == "Iron-air is early but real [1]."
    assert lines[2:] == [f"- Finding {i} [{i}]." for i in range(1, 6)]


async def test_too_few_findings_fall_back_to_the_body(cfg: QuarryConfig) -> None:
    reply = KeyFindingsPayload(bottom_line="Short.", findings=["Only one [1]."])
    section = await _write(reply, cfg)
    bullets = [line for line in section.markdown.splitlines() if line.startswith("- ")]
    assert bullets == [f"- Halvard Grid fact {i} holds [{i}]." for i in range(1, 7)]


async def test_a_failed_call_still_yields_the_section(cfg: QuarryConfig) -> None:
    section = await _write(ValueError("malformed JSON after retries"), cfg)
    assert section.markdown.startswith("- Halvard Grid fact 1 holds [1].")


async def test_cost_cap_always_propagates(cfg: QuarryConfig) -> None:
    with pytest.raises(CostCapExceeded):
        await _write(CostCapExceeded(5.01, 5.0), cfg)


def test_confidence_comes_from_independent_sites_and_tiers() -> None:
    index = _index()
    assert confidence_label([1, 2, 3], index, STRONG) == "High"
    assert confidence_label([2, 3, 6], index, STRONG) == "Medium"  # three sites, none strong
    assert confidence_label([4, 5], index, STRONG) == "Low"  # one site cited twice
    assert confidence_label([1], index, STRONG) == "Medium"  # one strong source


def test_labels_are_added_by_code_once() -> None:
    section = ReportSection(
        title=KEY_FINDINGS_TITLE,
        markdown="Bottom line [1].\n\n- A [1][2][3].\n- B [4][5].\n- No marker here.",
    )
    labeled = label_findings(section, _index(), STRONG)
    assert labeled.markdown.splitlines() == [
        "Bottom line [1].",
        "",
        "- **High confidence** (3 sites): A [1][2][3].",
        "- **Low confidence** (1 site): B [4][5].",
        "- No marker here.",
    ]
    assert label_findings(labeled, _index(), STRONG) == labeled  # resume-safe


def _plan() -> ResearchPlan:
    return ResearchPlan(
        topic="t",
        sub_questions=[
            SubQuestion(
                id=f"sq{i:02d}", question=f"Q{i}?", queries=["a", "b"], success_criterion="c"
            )
            for i in range(1, 9)
        ],
    )


def test_gaps_section_lists_only_uncovered_questions() -> None:
    gap = GapAnalysis(
        saturated=True,
        assessments=[
            CoverageAssessment(sub_question_id="sq01", covered=True),
            CoverageAssessment(sub_question_id="sq02", covered=False, missing="plant costs [4]."),
            CoverageAssessment(sub_question_id="sq03", covered=False, missing=None),
            CoverageAssessment(sub_question_id="sq99", covered=False, missing="ghost"),
        ],
    )
    section = gaps_section(_plan(), gap)
    assert section is not None and section.title == GAPS_TITLE
    assert section.markdown.splitlines() == [
        "- **Q2?** The sources reviewed did not establish this: plant costs.",
        "- **Q3?** The sources reviewed did not establish this.",
    ]


def test_full_coverage_says_so_and_no_gap_pass_means_no_section() -> None:
    covered = GapAnalysis(
        saturated=True, assessments=[CoverageAssessment(sub_question_id="sq01", covered=True)]
    )
    section = gaps_section(_plan(), covered)
    assert section is not None
    assert "every sub-question answered" in section.markdown
    assert gaps_section(_plan(), None) is None
