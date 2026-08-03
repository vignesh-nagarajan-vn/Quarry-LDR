"""PLAN stage: one Opus call decomposing the topic into sub-questions.

8 to 15 sub-questions, each with 2 to 4 seed queries and a success criterion
the gap analysis later scores coverage against.

Implemented in M8.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from quarry_ldr.config import QuarryConfig
from quarry_ldr.providers.anthropic_client import AnthropicProvider


class SubQuestion(BaseModel):
    id: str  # "sq01".."sq15"
    question: str
    queries: list[str] = Field(min_length=2, max_length=4)
    success_criterion: str


class ResearchPlan(BaseModel):
    topic: str
    sub_questions: list[SubQuestion] = Field(min_length=8, max_length=15)
    created_at: datetime | None = None

    @field_validator("sub_questions")
    @classmethod
    def _unique_ids(cls, v: list[SubQuestion]) -> list[SubQuestion]:
        ids = [sq.id for sq in v]
        if len(ids) != len(set(ids)):
            raise ValueError("sub-question ids must be unique")
        return v

    def all_queries(self) -> list[str]:
        return [q for sq in self.sub_questions for q in sq.queries]


async def make_plan(topic: str, provider: AnthropicProvider, cfg: QuarryConfig) -> ResearchPlan:
    """One models.plan call returning a validated ResearchPlan."""
    raise NotImplementedError
