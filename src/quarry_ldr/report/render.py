"""Markdown report assembly: sections, references, cost ledger, run manifest.

Implemented in M10 (basic form lands with M8's single pass).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from quarry_ldr.ledger import Ledger
from quarry_ldr.pipeline.synthesize import DraftReport
from quarry_ldr.report.citations import CitationIndex


class RunManifest(BaseModel):
    """Everything needed to understand how a report was produced."""

    run_id: str
    topic: str
    started_at: datetime
    finished_at: datetime
    iterations: int
    n_queries: int
    n_urls_fetched: int
    n_docs_extracted: int
    n_chunks: int
    n_chunks_after_dedup: int
    n_chunks_evidence: int
    models: dict[str, str] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


def render_report(
    draft: DraftReport,
    citations: CitationIndex,
    ledger: Ledger,
    manifest: RunManifest,
) -> str:
    """Assemble the full markdown document. Validates every citation marker
    resolves (CitationError otherwise) before returning."""
    raise NotImplementedError


def write_report(markdown: str, out_dir: Path, run_id: str) -> Path:
    """Write report markdown to out_dir/report-<run_id>.md and return the path."""
    raise NotImplementedError
