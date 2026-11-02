"""Markdown report assembly: sections, references, cost ledger, run manifest.

render_report validates every citation marker before returning: a broken
citation raises CitationError, and callers treat that as a failed run, not a
cosmetic issue.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from quarry_ldr.config import SourceSettings
from quarry_ldr.ingest.sources import SourceTier, rate_url, site_of
from quarry_ldr.ledger import Ledger
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
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
    n_conflicts: int = 0
    n_claims_checked: int = 0
    n_claims_rewritten: int = 0
    n_claims_dropped: int = 0
    n_figure_failures: int = 0
    n_support_rejections: int = 0
    n_claims_unjudged: int = 0
    n_uncited_attached: int = 0
    n_uncited_dropped: int = 0
    n_sections_removed: int = 0
    mean_entailment_score: float = 0.0
    models: dict[str, str] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


_TIER_LABELS = {
    SourceTier.PRIMARY: "Government, regulatory, and standards bodies",
    SourceTier.SCHOLARLY: "Journals, preprints, and universities",
    SourceTier.ESTABLISHED: "Reference works, major news, and trade press",
    SourceTier.GENERAL: "Company sites, blogs, and other web pages",
    SourceTier.LOW: "Market-report sites, machine-generated reports, and social media",
}


def sources_section(
    draft: DraftReport, citations: CitationIndex, settings: SourceSettings
) -> ReportSection:
    """What the report actually rests on, computed from its surviving markers."""
    urls = [
        citations.get(number).url
        for section in draft.sections
        for number in citations.numbers_in(section.markdown)
    ]
    if not urls:
        return ReportSection(title="Sources", markdown="This report cites no sources.")
    tiers = Counter(rate_url(url, settings.overrides) for url in urls)
    sites = Counter(site_of(url) for url in urls)
    lines = [f"This report makes {len(urls)} citations to {len(sites)} independent sites.", ""]
    for tier in SourceTier:
        if tiers[tier]:
            share = round(100 * tiers[tier] / len(urls))
            lines.append(f"- **{_TIER_LABELS[tier]}:** {share} percent of citations")
    site, count = sites.most_common(1)[0]
    if len(sites) == 1:
        lines += ["", f"Caution: every citation comes from one site, {site}."]
    elif count / len(urls) > settings.max_site_share:
        lines += [
            "",
            f"Caution: {round(100 * count / len(urls))} percent of citations come from one "
            f"site, {site}. Claims resting on it have less independent support.",
        ]
    return ReportSection(title="Sources", markdown="\n".join(lines))


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
        citations.references_markdown(body),
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
