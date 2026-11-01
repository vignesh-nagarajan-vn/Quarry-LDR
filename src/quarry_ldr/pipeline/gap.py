"""GAP stage: Sonnet compares evidence coverage against the plan's success
criteria and either emits new queries or declares saturation.

Runs every iteration, so it uses the cheap model.

Implemented in M9.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig
from quarry_ldr.ingest.chunk import HeuristicTokenCounter
from quarry_ldr.pipeline.plan import ResearchPlan
from quarry_ldr.pipeline.triage import TriagedChunk
from quarry_ldr.providers.base import Provider
from quarry_ldr.report.claims import extract_figures, named_entities


class CoverageAssessment(BaseModel):
    sub_question_id: str
    covered: bool
    # What evidence is still absent. None and "" both mean "nothing":
    # Haiku emits an explicit null for covered sub-questions where Sonnet
    # emits an empty string, and rejecting the null cost every assisted
    # gap call a schema-retry round trip before this widened.
    missing: str | None = ""


class GapAnalysis(BaseModel):
    saturated: bool
    assessments: list[CoverageAssessment] = Field(default_factory=list)
    new_queries: list[str] = Field(default_factory=list)
    rationale: str = ""


GAP_SYSTEM = (
    "You audit research coverage. For each sub-question, judge whether the "
    "collected claims satisfy its success criterion and pin down its named "
    "specifics. Generalities with no names, dates, or figures do not count as "
    "covered. Emit new, DIFFERENT web "
    "search queries only for uncovered sub-questions (never repeat earlier "
    "angles). Declare saturation when further searching is unlikely to add "
    'evidence that changes the report. Reply with JSON only: {"saturated": '
    'true/false, "assessments": [{"sub_question_id": "...", "covered": '
    'true/false, "missing": "..."}], "new_queries": ["..."], "rationale": "..."}'
)

_MAX_CLAIMS_PER_SQ = 40
_DIGEST_SHRINK_STEPS = (40, 20, 10, 5, 3, 1)
_LAST_RESORT_CLAIM_CHARS = 160
# A sub-question with fewer specifics than this gets code-issued follow-ups.
_MIN_NAMES = 2
_MIN_FIGURES = 1
_MAX_FOLLOWUPS = 4


def specifics_found(claims: list[str]) -> tuple[int, int]:
    """Distinct names and figures across claims."""
    joined = " ".join(claims)
    return len(named_entities(joined)), len(extract_figures(joined))


def specificity_followups(
    plan: ResearchPlan, evidence: list[TriagedChunk], asked: set[str]
) -> list[str]:
    """Queries for sub-questions whose evidence stays vague, decided by code.

    A model can declare saturation over claims that name nobody; this check
    keeps searching anyway, bounded per pass, and never repeats a query.
    """
    claims_by_sq: dict[str, list[str]] = {}
    for item in evidence:
        claims_by_sq.setdefault(item.sub_question_id, []).append(item.verdict.claim)
    followups: list[str] = []
    for sub_question in plan.sub_questions:
        names, figures = specifics_found(claims_by_sq.get(sub_question.id, []))
        if names >= _MIN_NAMES and figures >= _MIN_FIGURES:
            continue
        focus = sub_question.must_pin_down or "named examples and measured figures"
        query = f"{plan.topic} {focus}"
        if query not in asked and query not in followups:
            followups.append(query)
        if len(followups) == _MAX_FOLLOWUPS:
            break
    return followups


def coverage_digest(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    max_claims_per_sq: int = _MAX_CLAIMS_PER_SQ,
    truncate_chars: int | None = None,
) -> str:
    """Compact per-sub-question digest: criterion plus collected claims."""
    claims_by_sq: dict[str, list[str]] = {}
    for item in evidence:
        claims_by_sq.setdefault(item.sub_question_id, []).append(item.verdict.claim)
    lines: list[str] = []
    for sub_question in plan.sub_questions:
        claims = claims_by_sq.get(sub_question.id, [])[:max_claims_per_sq]
        if truncate_chars is not None:
            claims = [claim[:truncate_chars] for claim in claims]
        lines.append(f"{sub_question.id}: {sub_question.question}")
        lines.append(f"  success criterion: {sub_question.success_criterion}")
        if sub_question.must_pin_down:
            lines.append(f"  must pin down: {sub_question.must_pin_down}")
        names, figures = specifics_found(claims_by_sq.get(sub_question.id, []))
        lines.append(f"  specifics found: {names} names, {figures} figures")
        if claims:
            lines.extend(f"  claim: {claim}" for claim in claims)
        else:
            lines.append("  (no evidence collected)")
        lines.append("")
    return "\n".join(lines)


def budgeted_digest(plan: ResearchPlan, evidence: list[TriagedChunk], budget_tokens: int) -> str:
    """The digest shrunk until it fits ``budget_tokens`` heuristic tokens.

    A 14-sub-question plan at 40 claims each overflows the triage server's
    8K context (measured live, M16); the digest steps down through fewer
    claims per sub-question and, as a last resort, hard-truncates claims.
    """
    counter = HeuristicTokenCounter()
    for cap in _DIGEST_SHRINK_STEPS:
        digest = coverage_digest(plan, evidence, max_claims_per_sq=cap)
        if counter.count(digest) <= budget_tokens:
            return digest
    return coverage_digest(
        plan, evidence, max_claims_per_sq=1, truncate_chars=_LAST_RESORT_CLAIM_CHARS
    )


async def analyze_gaps(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    provider: Provider,
    cfg: QuarryConfig,
    iteration: int,
    model: str | None = None,
    digest_budget_tokens: int | None = None,
    max_tokens: int = 4096,
) -> GapAnalysis:
    """One gap call. saturated=True or an empty new_queries list ends the
    loop; the orchestrator also enforces run.max_iterations regardless.

    ``model`` overrides ``cfg.models.gap`` (the orchestrator passes
    models.assisted in assisted mode). ``digest_budget_tokens`` bounds the
    digest for context-limited local backends. Stage code stays engine-agnostic.
    """
    digest = (
        budgeted_digest(plan, evidence, digest_budget_tokens)
        if digest_budget_tokens is not None
        else coverage_digest(plan, evidence)
    )
    prompt = f"Topic: {plan.topic}\nCompleted search iterations so far: {iteration + 1}\n\n{digest}"
    return await provider.complete_typed(
        model=model if model is not None else cfg.models.gap,
        system=GAP_SYSTEM,
        prompt=prompt,
        schema=GapAnalysis,
        # claude-sonnet-5 thinks inside this budget; 2000 truncated a live
        # gap call at exactly max_tokens and paid a retry.
        max_tokens=max_tokens,
        stage="gap",
        iteration=iteration,
    )
