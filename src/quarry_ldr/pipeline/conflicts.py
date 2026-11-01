"""CONFLICTS stage: find figures that different sources report differently.

The local 4B proposes conflicts; code decides. A proposal survives only if
both values are copied from their claims, the claims come from different
sites, the units match, and the numbers truly differ. Runs in every engine
mode at no API cost, so a report cannot ship 46 percent beside 58 percent
without saying so.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig
from quarry_ldr.gpu.local_llm import LocalLLM
from quarry_ldr.ingest.sources import site_of
from quarry_ldr.logging import get_logger
from quarry_ldr.pipeline.plan import ResearchPlan, SubQuestion
from quarry_ldr.pipeline.triage import TriagedChunk
from quarry_ldr.report.citations import CitationIndex
from quarry_ldr.report.claims import extract_figures, strip_markers

logger = get_logger(component="conflicts")

_MAX_CLAIMS_PER_SQ = 30
_MAX_CLAIM_CHARS = 300
_MAX_TOKENS = 512
_MAX_MEASURE_CHARS = 80
_MAX_CONFLICTS = 8
# Sources rounding the same number are not disagreeing.
_RELATIVE_TOLERANCE = 0.05
_QUALIFIERS = re.compile(
    r"\b(about|approximately|around|roughly|nearly|over|under|above|below|up to|at least|"
    r"at most|more than|less than|as much as|as low as|as high as|an?|the|of)\b"
)
_YEAR_RANGE = (1900.0, 2100.0)


class ConflictSide(BaseModel):
    chunk_id: str
    claim: str
    value: str


class Conflict(BaseModel):
    sub_question_id: str
    measure: str
    a: ConflictSide
    b: ConflictSide


class _Proposed(BaseModel):
    measure: str
    claim_a: int
    value_a: str
    claim_b: int
    value_b: str


class _ProposedConflicts(BaseModel):
    conflicts: list[_Proposed] = Field(default_factory=list)


CONFLICT_PROMPT = """You compare factual claims from different sources about one research question.

Question: {question}

Claims:
{claims}

Find pairs of claims that give different values for the same measure of the same thing under the
same conditions. Ignore pairs that differ because of theoretical versus practical figures,
different years, places, products, or scenarios.

Reply with JSON only: {{"conflicts": [{{"measure": "short name of what is measured", "claim_a": 1,
"value_a": "the value with its unit, copied exactly from claim 1", "claim_b": 2, "value_b": "the
value with its unit, copied exactly from claim 2"}}]}}. Reply {{"conflicts": []}} if there are none.
"""


def comparable_claims(evidence: list[TriagedChunk]) -> list[TriagedChunk]:
    """Claims with figures, one per chunk: only these can disagree numerically."""
    seen: set[str] = set()
    claims: list[TriagedChunk] = []
    for item in evidence:
        if item.chunk.chunk_id in seen or not extract_figures(item.verdict.claim):
            continue
        seen.add(item.chunk.chunk_id)
        claims.append(item)
    return claims[:_MAX_CLAIMS_PER_SQ]


def _interval(value: str) -> tuple[float, float]:
    numbers = [float(figure) for figure in extract_figures(value)]
    return min(numbers), max(numbers)


def _unit(value: str) -> str:
    text = value.lower().replace("percent", "%")
    text = re.sub(r"\d[\d,]*(?:\.\d+)?", " ", text)
    text = _QUALIFIERS.sub(" ", text)
    return re.sub(r"[\s\-\u2013~\u2248.,]+", "", text)


def _values_differ(value_a: str, value_b: str) -> bool:
    low_a, high_a = _interval(value_a)
    low_b, high_b = _interval(value_b)
    if low_a <= high_b and low_b <= high_a:
        return False  # overlapping ranges agree
    gap = max(low_a, low_b) - min(high_a, high_b)
    return gap / max(abs(high_a), abs(high_b), 1e-9) > _RELATIVE_TOLERANCE


def _looks_like_years(value: str) -> bool:
    low, high = _interval(value)
    return not _unit(value) and _YEAR_RANGE[0] <= low and high <= _YEAR_RANGE[1]


def _accept(
    proposal: _Proposed, claims: list[TriagedChunk], sub_question_id: str
) -> Conflict | None:
    """Code's decision on one model proposal; None rejects it."""
    if not (1 <= proposal.claim_a <= len(claims) and 1 <= proposal.claim_b <= len(claims)):
        return None
    item_a, item_b = claims[proposal.claim_a - 1], claims[proposal.claim_b - 1]
    if site_of(item_a.chunk.url) == site_of(item_b.chunk.url):
        return None
    value_a, value_b = proposal.value_a.strip(), proposal.value_b.strip()
    for value, item in ((value_a, item_a), (value_b, item_b)):
        if not extract_figures(value) or value.lower() not in item.verdict.claim.lower():
            return None
    if _unit(value_a) != _unit(value_b):
        return None
    if _looks_like_years(value_a) or _looks_like_years(value_b):
        return None
    if not _values_differ(value_a, value_b):
        return None
    measure = strip_markers(proposal.measure).replace("*", "").strip()[:_MAX_MEASURE_CHARS]
    if not measure:
        return None
    return Conflict(
        sub_question_id=sub_question_id,
        measure=measure,
        a=ConflictSide(chunk_id=item_a.chunk.chunk_id, claim=item_a.verdict.claim, value=value_a),
        b=ConflictSide(chunk_id=item_b.chunk.chunk_id, claim=item_b.verdict.claim, value=value_b),
    )


async def _conflicts_for(
    sub_question: SubQuestion, evidence: list[TriagedChunk], llm: LocalLLM, cfg: QuarryConfig
) -> list[Conflict]:
    claims = comparable_claims(evidence)
    if len({site_of(item.chunk.url) for item in claims}) < 2:
        return []  # nothing to compare: no model call
    listing = "\n".join(
        f"{number}. {item.verdict.claim[:_MAX_CLAIM_CHARS]}"
        for number, item in enumerate(claims, start=1)
    )
    prompt = CONFLICT_PROMPT.format(question=sub_question.question, claims=listing)
    try:
        payload = await llm.complete_typed(
            prompt, _ProposedConflicts, max_tokens=_MAX_TOKENS, max_retries=cfg.triage.max_retries
        )
    except ValueError:
        logger.warning("conflicts_malformed", sub_question=sub_question.id)
        return []
    accepted = (_accept(proposal, claims, sub_question.id) for proposal in payload.conflicts)
    return [conflict for conflict in accepted if conflict is not None]


async def find_conflicts(
    plan: ResearchPlan, evidence: list[TriagedChunk], llm: LocalLLM, cfg: QuarryConfig
) -> list[Conflict]:
    """Conflicts within each sub-question, deduplicated and capped for the report."""
    by_sq: dict[str, list[TriagedChunk]] = {}
    for item in evidence:
        by_sq.setdefault(item.sub_question_id, []).append(item)
    found: dict[frozenset[tuple[str, str]], Conflict] = {}
    for sub_question in plan.sub_questions:
        for conflict in await _conflicts_for(
            sub_question, by_sq.get(sub_question.id, []), llm, cfg
        ):
            key = frozenset(
                {(conflict.a.chunk_id, conflict.a.value), (conflict.b.chunk_id, conflict.b.value)}
            )
            found.setdefault(key, conflict)
    conflicts = list(found.values())[:_MAX_CONFLICTS]
    logger.info("conflicts_found", n=len(conflicts))
    return conflicts


def _numbers(conflict: Conflict, citations: CitationIndex) -> tuple[int, int] | None:
    number_a = citations.number_for(conflict.a.chunk_id)
    number_b = citations.number_for(conflict.b.chunk_id)
    return None if number_a is None or number_b is None else (number_a, number_b)


def conflict_notes(
    conflicts: list[Conflict], citations: CitationIndex, sub_question_ids: list[str] | None = None
) -> str:
    """Writer briefing, empty when nothing applies. Deterministic, because the
    premium path keeps it inside its byte-stable corpus."""
    lines: list[str] = []
    for conflict in conflicts:
        if sub_question_ids is not None and conflict.sub_question_id not in sub_question_ids:
            continue
        numbers = _numbers(conflict, citations)
        if numbers is not None:
            lines.append(
                f"- {conflict.measure}: [{numbers[0]}] reports {conflict.a.value}; "
                f"[{numbers[1]}] reports {conflict.b.value}"
            )
    if not lines:
        return ""
    return "KNOWN DISAGREEMENTS BETWEEN SOURCES (state both sides)\n" + "\n".join(lines)


def disagreement_markdown(conflicts: list[Conflict], citations: CitationIndex) -> str:
    """The report's disagreement list, one checkable bullet per conflict."""
    lines: list[str] = []
    for conflict in conflicts:
        numbers = _numbers(conflict, citations)
        if numbers is None:
            continue
        claim_a = conflict.a.claim.strip().rstrip(".")
        claim_b = conflict.b.claim.strip().rstrip(".")
        lines.append(
            f"- **{conflict.measure}:** {claim_a} [{numbers[0]}], while another source "
            f"reports: {claim_b} [{numbers[1]}]."
        )
    return "\n".join(lines)
