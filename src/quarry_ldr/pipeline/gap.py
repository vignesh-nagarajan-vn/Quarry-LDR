"""GAP stage: Sonnet compares evidence coverage against the plan's success
criteria and either emits new queries or declares saturation.

Runs every iteration, so it uses the cheap model.

Implemented in M9.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig
from quarry_ldr.pipeline.plan import ResearchPlan
from quarry_ldr.pipeline.triage import TriagedChunk
from quarry_ldr.providers.base import Provider


class CoverageAssessment(BaseModel):
    sub_question_id: str
    covered: bool
    missing: str = ""  # what evidence is still absent, empty when covered


class GapAnalysis(BaseModel):
    saturated: bool
    assessments: list[CoverageAssessment] = Field(default_factory=list)
    new_queries: list[str] = Field(default_factory=list)
    rationale: str = ""


GAP_SYSTEM = (
    "You audit research coverage. For each sub-question, judge whether the "
    "collected claims satisfy its success criterion. Emit new, DIFFERENT web "
    "search queries only for uncovered sub-questions (never repeat earlier "
    "angles). Declare saturation when further searching is unlikely to add "
    'evidence that changes the report. Reply with JSON only: {"saturated": '
    'true/false, "assessments": [{"sub_question_id": "...", "covered": '
    'true/false, "missing": "..."}], "new_queries": ["..."], "rationale": "..."}'
)

_MAX_CLAIMS_PER_SQ = 40


def coverage_digest(plan: ResearchPlan, evidence: list[TriagedChunk]) -> str:
    """Compact per-sub-question digest: criterion plus collected claims."""
    claims_by_sq: dict[str, list[str]] = {}
    for item in evidence:
        claims_by_sq.setdefault(item.sub_question_id, []).append(item.verdict.claim)
    lines: list[str] = []
    for sub_question in plan.sub_questions:
        claims = claims_by_sq.get(sub_question.id, [])[:_MAX_CLAIMS_PER_SQ]
        lines.append(f"{sub_question.id}: {sub_question.question}")
        lines.append(f"  success criterion: {sub_question.success_criterion}")
        if claims:
            lines.extend(f"  claim: {claim}" for claim in claims)
        else:
            lines.append("  (no evidence collected)")
        lines.append("")
    return "\n".join(lines)


async def analyze_gaps(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    provider: Provider,
    cfg: QuarryConfig,
    iteration: int,
    model: str | None = None,
) -> GapAnalysis:
    """One gap call. saturated=True or an empty new_queries list ends the
    loop; the orchestrator also enforces run.max_iterations regardless.

    ``model`` overrides ``cfg.models.gap`` (the orchestrator passes
    models.assisted in assisted mode); stage code stays engine-agnostic.
    """
    prompt = (
        f"Topic: {plan.topic}\n"
        f"Completed search iterations so far: {iteration + 1}\n\n"
        f"{coverage_digest(plan, evidence)}"
    )
    return await provider.complete_typed(
        model=model if model is not None else cfg.models.gap,
        system=GAP_SYSTEM,
        prompt=prompt,
        schema=GapAnalysis,
        # claude-sonnet-5 thinks inside this budget; 2000 truncated a live
        # gap call at exactly max_tokens and paid a retry.
        max_tokens=4096,
        stage="gap",
        iteration=iteration,
    )
