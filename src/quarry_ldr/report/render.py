"""Markdown report assembly: sections, references, cost ledger, run manifest.

render_report validates every citation marker before returning: a broken
citation raises CitationError, and callers treat that as a failed run, not a
cosmetic issue.
"""

from __future__ import annotations

import json
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
    body_parts: list[str] = [f"# {draft.topic}", ""]
    for section in draft.sections:
        body_parts.append(f"## {section.title}")
        body_parts.append("")
        body_parts.append(section.markdown)
        body_parts.append("")
    body = "\n".join(body_parts)
    citations.validate_markdown(body)

    manifest_block = json.dumps(
        manifest.model_dump(mode="json", exclude={"config_snapshot"}),
        indent=2,
        sort_keys=True,
    )
    parts = [
        body,
        citations.references_markdown(),
        "",
        ledger.to_markdown(),
        "",
        "## Run manifest",
        "",
        "```json",
        manifest_block,
        "```",
        "",
    ]
    return "\n".join(parts)


def write_report(markdown: str, out_dir: Path, run_id: str) -> Path:
    """Write report markdown to out_dir/report-<run_id>.md and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report-{run_id}.md"
    path.write_text(markdown, encoding="utf-8", newline="\n")
    return path
