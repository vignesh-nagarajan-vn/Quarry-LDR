"""SYNTHESIZE stage: Opus writes the report section by section over one
prompt-cached evidence corpus.

The corpus is built once, hashed once, and every section call sends the byte
identical prefix with a 1h cache_control breakpoint; the provider raises
CachePrefixError on any drift rather than silently paying full price.

Implemented in M8.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quarry_ldr.config import QuarryConfig
from quarry_ldr.pipeline.plan import ResearchPlan
from quarry_ldr.pipeline.triage import TriagedChunk
from quarry_ldr.providers.anthropic_client import AnthropicProvider
from quarry_ldr.report.citations import CitationIndex


class SectionBrief(BaseModel):
    """One planned section of the report."""

    title: str
    instructions: str
    sub_question_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    title: str
    markdown: str  # cites evidence as [n] markers resolved by CitationIndex


class DraftReport(BaseModel):
    topic: str
    sections: list[ReportSection] = Field(default_factory=list)
    corpus_hash: str = ""


def build_evidence_corpus(evidence: list[TriagedChunk], citations: CitationIndex) -> str:
    """Deterministic corpus: evidence grouped by sub-question, each chunk
    tagged with its citation number. Byte-stable across section calls."""
    raise NotImplementedError


def plan_sections(plan: ResearchPlan, cfg: QuarryConfig) -> list[SectionBrief]:
    """Derive section briefs from the research plan (min/max from config)."""
    raise NotImplementedError


async def synthesize(
    plan: ResearchPlan,
    evidence: list[TriagedChunk],
    provider: AnthropicProvider,
    cfg: QuarryConfig,
    citations: CitationIndex,
) -> DraftReport:
    """Section-by-section models.synthesize calls over the cached corpus."""
    raise NotImplementedError
