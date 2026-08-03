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
from quarry_ldr.providers.anthropic_client import AnthropicProvider


class CoverageAssessment(BaseModel):
    sub_question_id: str
    covered: bool
    missing: str = ""  # what evidence is still absent, empty when covered


class GapAnalysis(BaseModel):
    saturated: bool
    assessments: list[CoverageAssessment] = Field(default_factory=list)
    new_queries: list[str] = Field(default_factory=list)
    rationale: str = ""


async def analyze_gaps(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    provider: AnthropicProvider,
    cfg: QuarryConfig,
    iteration: int,
) -> GapAnalysis:
    """One models.gap call. saturated=True or an empty new_queries list ends
    the loop; the orchestrator also enforces run.max_iterations regardless."""
    raise NotImplementedError
