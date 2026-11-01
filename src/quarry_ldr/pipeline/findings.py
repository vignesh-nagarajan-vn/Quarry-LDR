"""Key findings and the evidence gaps, the report's opening and closing sections.

Key findings are written last, from the cleaned body, by the engine's writer.
Code keeps a bullet only if it cites a marker the body already uses, falls
back to the body's own cited sentences when the model fails, and assigns
every confidence label at RENDER from the surviving markers: models never
label their own confidence. "What we could not establish" is built by code
from the final GAP pass.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig, SourceSettings
from quarry_ldr.ingest.sources import STRONG_TIERS, rate_url, site_of
from quarry_ldr.ledger import CostCapExceeded
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.gap import GapAnalysis
from quarry_ldr.pipeline.plan import ResearchPlan
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.providers.base import Provider
from quarry_ldr.report.citations import CITATION_MARKER, CitationIndex
from quarry_ldr.report.claims import split_sentences, strip_markers

logger = get_logger(component="findings")

KEY_FINDINGS_TITLE = "Key findings"
GAPS_TITLE = "What we could not establish"
MIN_FINDINGS = 5
MAX_FINDINGS = 8
_MAX_OPEN_QUERIES = 5
_BULLET = re.compile(r"^- (?!\*\*(?:High|Medium|Low) confidence)(.*)$")


class KeyFindingsPayload(BaseModel):
    """What the writer returns; code decides what survives."""

    bottom_line: str
    findings: list[str] = Field(min_length=1, max_length=12)


FINDINGS_SYSTEM = (
    "You write the opening of a research report from its finished body. Give a "
    "bottom line of two or three sentences, then 5 to 8 key findings. Each finding "
    "is one sentence stating a concrete result, with names, dates, and figures "
    "where the body gives them, and ends with the [n] markers the body uses for "
    "that result. Use only facts and markers that appear in the body. Do not "
    "rate confidence. "
    'Reply with JSON only: {"bottom_line": "...", "findings": ["...", "..."]}.'
)


def _body_text(draft: DraftReport) -> str:
    return "\n\n".join(f"## {s.title}\n\n{s.markdown}" for s in draft.sections)


def _keep_known(text: str, allowed: set[int]) -> str:
    """Drop markers the body does not use, closing the gap they leave."""
    kept = CITATION_MARKER.sub(lambda m: m.group(0) if int(m.group(1)) in allowed else "", text)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", kept).strip()


def _fallback_findings(draft: DraftReport) -> list[str]:
    """The first cited sentence of each body section, in order."""
    found: list[str] = []
    for section in draft.sections:
        for sentence in split_sentences(section.markdown.replace("\n", " ")):
            if CITATION_MARKER.search(sentence) and not sentence.startswith(("#", "|")):
                found.append(sentence.lstrip("-* ").strip())
                break
        if len(found) == MAX_FINDINGS:
            break
    return found


def findings_markdown(bottom_line: str, findings: list[str]) -> str:
    lines = [bottom_line.strip(), ""] if bottom_line.strip() else []
    lines.extend(f"- {finding}" for finding in findings)
    return "\n".join(lines)


async def write_key_findings(
    draft: DraftReport,
    provider: Provider,
    cfg: QuarryConfig,
    citations: CitationIndex,
    max_tokens: int,
) -> ReportSection:
    """One writer call over the cleaned body, guarded by code.

    A finding survives only with at least one marker the body already uses.
    Fewer than five survivors, or any failure short of the cost cap, falls
    back to the body's own cited sentences, so the section always exists.
    """
    body = _body_text(draft)
    allowed = set(citations.numbers_in(body))
    bottom_line = ""
    findings: list[str] = []
    try:
        payload = await provider.complete_typed(
            model=cfg.models.synthesize,
            system=FINDINGS_SYSTEM,
            prompt=f"Report topic: {draft.topic}\n\n{body}",
            schema=KeyFindingsPayload,
            max_tokens=max_tokens,
            stage="findings",
        )
        bottom_line = _keep_known(payload.bottom_line, allowed)
        for raw in payload.findings:
            finding = _keep_known(raw.lstrip("-* ").replace("\n", " "), allowed)
            if CITATION_MARKER.search(finding) and finding not in findings:
                findings.append(finding)
    except CostCapExceeded:
        raise
    except Exception as exc:  # the section must exist; the fallback is code
        logger.warning("findings_failed", error=str(exc)[:200])
    if len(findings) < MIN_FINDINGS:
        logger.warning("findings_fallback", n_model=len(findings))
        findings = _fallback_findings(draft)
    return ReportSection(
        title=KEY_FINDINGS_TITLE,
        markdown=findings_markdown(bottom_line, findings[:MAX_FINDINGS]),
    )


def confidence_label(numbers: list[int], citations: CitationIndex, settings: SourceSettings) -> str:
    """High: three or more independent sites including a primary or scholarly
    one. Medium: two or more sites, or one strong source. Low: otherwise."""
    urls = [citations.get(n).url for n in numbers]
    sites = {site_of(url) for url in urls}
    strong = any(rate_url(url, settings.overrides) in STRONG_TIERS for url in urls)
    if len(sites) >= 3 and strong:
        return "High"
    if len(sites) >= 2 or strong:
        return "Medium"
    return "Low"


def label_findings(
    section: ReportSection, citations: CitationIndex, settings: SourceSettings
) -> ReportSection:
    """Prefix each finding bullet with a code-assigned confidence label."""
    lines: list[str] = []
    for line in section.markdown.split("\n"):
        match = _BULLET.match(line)
        numbers = citations.numbers_in(line)
        if match and numbers:
            label = confidence_label(numbers, citations, settings)
            n_sites = len({site_of(citations.get(n).url) for n in numbers})
            plural = "site" if n_sites == 1 else "sites"
            line = f"- **{label} confidence** ({n_sites} {plural}): {match.group(1)}"
        lines.append(line)
    return section.model_copy(update={"markdown": "\n".join(lines)})


def gaps_section(plan: ResearchPlan, gap: GapAnalysis | None) -> ReportSection | None:
    """What the final GAP pass left open; None only when no GAP ran.

    Besides uncovered sub-questions, this lists the searches an unsaturated
    final pass still asked for. The live 4B marked every sub-question covered
    while asking for pilot efficiency data no source had, so "covered" alone
    cannot justify telling the reader that everything was answered.
    """
    if gap is None:
        return None
    by_id = {sq.id: sq for sq in plan.sub_questions}
    lines: list[str] = []
    for assessment in gap.assessments:
        sub_question = by_id.get(assessment.sub_question_id)
        if sub_question is None or assessment.covered:
            continue
        missing = strip_markers(assessment.missing or "").rstrip(".")
        detail = f": {missing}." if missing else "."
        lines.append(
            f"- **{sub_question.question}** The sources reviewed did not establish this{detail}"
        )
    if not gap.saturated:
        for query in gap.new_queries[:_MAX_OPEN_QUERIES]:
            wanted = strip_markers(query).strip().rstrip(".")
            if wanted:
                lines.append(f"- **Still open after the final search:** {wanted}.")
    if not lines:
        lines = [
            "The final coverage check found every sub-question answered by the sources reviewed."
        ]
    return ReportSection(title=GAPS_TITLE, markdown="\n".join(lines))
