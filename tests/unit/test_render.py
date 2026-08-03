"""Report assembly: sections, references, cost ledger, run manifest, and the
hard failure on a broken citation marker."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quarry_ldr.ingest.chunk import Chunk
from quarry_ldr.ledger import Ledger, TokenUsage
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.report.citations import CitationError, CitationIndex
from quarry_ldr.report.render import RunManifest, render_report, write_report

RUN_DATE = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def make_chunk(chunk_id: str, url: str = "https://vesterholm-times.example/a") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        url=url,
        doc_title="Test doc",
        heading_path=["Background"],
        text="passage text",
        token_count=10,
        position=0,
        start_char=100,
        end_char=200,
    )


def make_manifest(**overrides: object) -> RunManifest:
    fields: dict[str, object] = {
        "run_id": "run-abc123",
        "topic": "the vesterholm sand battery pilot",
        "started_at": RUN_DATE,
        "finished_at": RUN_DATE,
        "iterations": 2,
        "n_queries": 4,
        "n_urls_fetched": 8,
        "n_docs_extracted": 6,
        "n_chunks": 20,
        "n_chunks_after_dedup": 15,
        "n_chunks_evidence": 5,
        "models": {"synthesize": "claude-opus-5"},
        "config_snapshot": {"run": {"cost_cap_usd": 5.0}, "anthropic_api_key": None},
    }
    fields.update(overrides)
    return RunManifest(**fields)  # type: ignore[arg-type]


def make_draft(sections: list[ReportSection] | None = None) -> DraftReport:
    if sections is None:
        sections = [
            ReportSection(title="Overview", markdown="A claim about the pilot [1]."),
            ReportSection(title="Conclusions", markdown="It works well [2]."),
        ]
    return DraftReport(topic="the vesterholm sand battery pilot", sections=sections)


def make_ledger() -> Ledger:
    ledger = Ledger(cost_cap_usd=5.0)
    ledger.record(
        "claude-opus-5",
        TokenUsage(input_tokens=1000, output_tokens=200),
        stage="synthesize",
        on=RUN_DATE.date(),
    )
    return ledger


def test_full_render_contains_all_sections() -> None:
    citations = CitationIndex()
    citations.add(make_chunk("aaa"))
    citations.add(make_chunk("bbb", url="https://vesterholm-times.example/b"))
    draft = make_draft()
    ledger = make_ledger()
    manifest = make_manifest()

    markdown = render_report(draft, citations, ledger, manifest)

    assert markdown.startswith("# the vesterholm sand battery pilot")
    assert "## Overview" in markdown
    assert "## Conclusions" in markdown
    assert "## References" in markdown
    assert "[1]" in markdown and "[2]" in markdown
    assert "## Cost ledger" in markdown
    assert "## Run manifest" in markdown
    assert "```json" in markdown

    # The manifest block is valid, well-formed JSON.
    fence_start = markdown.index("```json") + len("```json")
    fence_end = markdown.index("```", fence_start)
    manifest_json = json.loads(markdown[fence_start:fence_end])
    assert manifest_json["run_id"] == "run-abc123"
    assert manifest_json["n_chunks_evidence"] == 5


def test_broken_citation_raises_citation_error() -> None:
    citations = CitationIndex()
    citations.add(make_chunk("aaa"))
    draft = make_draft(
        sections=[
            ReportSection(title="Overview", markdown="A claim citing [7] which does not exist.")
        ]
    )
    ledger = make_ledger()
    manifest = make_manifest()

    with pytest.raises(CitationError):
        render_report(draft, citations, ledger, manifest)


def test_manifest_json_excludes_config_snapshot() -> None:
    citations = CitationIndex()
    citations.add(make_chunk("aaa"))
    draft = make_draft(sections=[ReportSection(title="Overview", markdown="A claim [1].")])
    ledger = make_ledger()
    manifest = make_manifest()

    markdown = render_report(draft, citations, ledger, manifest)

    fence_start = markdown.index("```json") + len("```json")
    fence_end = markdown.index("```", fence_start)
    manifest_json = json.loads(markdown[fence_start:fence_end])
    assert "config_snapshot" not in manifest_json
    # The raw block text must not leak the snapshot either.
    assert "cost_cap_usd" not in markdown[fence_start:fence_end]


def test_empty_ledger_render_still_works() -> None:
    citations = CitationIndex()
    citations.add(make_chunk("aaa"))
    draft = make_draft(sections=[ReportSection(title="Overview", markdown="A claim [1].")])
    ledger = Ledger()
    manifest = make_manifest()

    markdown = render_report(draft, citations, ledger, manifest)

    assert "## Cost ledger" in markdown
    assert "Total: $0.0000" in markdown


def test_render_with_no_citations_and_no_markers() -> None:
    """An empty CitationIndex and a section with no [n] markers must render fine."""
    citations = CitationIndex()
    draft = make_draft(
        sections=[ReportSection(title="Overview", markdown="No citations needed here.")]
    )
    ledger = Ledger()
    manifest = make_manifest(n_chunks_evidence=0)

    markdown = render_report(draft, citations, ledger, manifest)

    assert "## References" in markdown
    assert "## Overview" in markdown


def test_write_report_writes_lf_file_named_by_run_id(tmp_path: Path) -> None:
    # write_report must not let Windows text-mode translate "\n" to "\r\n".
    markdown = "# Title\n\nSome body\n"
    path = write_report(markdown, tmp_path, "run-abc123")

    assert path == tmp_path / "report-run-abc123.md"
    assert path.is_file()

    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") >= 2


def test_write_report_creates_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "nested" / "reports"
    path = write_report("# Title\n", out_dir, "run-xyz")
    assert path.is_file()
    assert path.parent == out_dir
