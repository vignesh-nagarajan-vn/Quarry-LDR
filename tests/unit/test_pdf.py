"""PDF deliverable: Typst escaping, markdown conversion, offline compile."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quarry_ldr.ingest.chunk import Chunk, make_chunk_id
from quarry_ldr.ledger import Ledger
from quarry_ldr.pipeline.synthesize import DraftReport, ReportSection
from quarry_ldr.report.citations import CitationIndex
from quarry_ldr.report.pdf import (
    _TEMPLATE_PATH,
    _escape_typst,
    _inline,
    _markdown_to_typst,
    render_pdf,
)
from quarry_ldr.report.render import RunManifest

TINY_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
    '<rect width="10" height="10" fill="#b45309"/></svg>'
)


def make_manifest() -> RunManifest:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    return RunManifest(
        run_id="run1234",
        topic="fixture topic",
        started_at=now,
        finished_at=now,
        iterations=2,
        n_queries=8,
        n_urls_fetched=30,
        n_docs_extracted=25,
        n_chunks=400,
        n_chunks_after_dedup=350,
        n_chunks_evidence=120,
        n_claims_checked=40,
        n_claims_rewritten=2,
        n_claims_dropped=1,
        models={"engine": "local"},
    )


def make_citations(n: int = 2) -> CitationIndex:
    citations = CitationIndex()
    for i in range(n):
        url = "https://vesterholm-times.example/2024/article"
        citations.add(
            Chunk(
                chunk_id=make_chunk_id(url, i),
                url=url,
                doc_title="A #titled [doc] with *specials*",
                heading_path=["Background"],
                text="chunk text",
                token_count=3,
                position=i,
                start_char=0,
                end_char=10,
            )
        )
    return citations


def test_template_ships_with_the_package() -> None:
    assert _TEMPLATE_PATH.is_file()
    assert "quarry-report" in _TEMPLATE_PATH.read_text(encoding="utf-8")


def test_escape_covers_typst_markup() -> None:
    escaped = _escape_typst(r"#call $math [box] *bold* _em_ `raw` <tag> @ref ~ back\slash")
    for token in ("\\#", "\\$", "\\[", "\\]", "\\*", "\\_", "\\`", "\\<", "\\>", "\\@", "\\~"):
        assert token in escaped
    assert "back\\\\slash" in escaped


def test_inline_preserves_bold_and_citation_markers() -> None:
    converted = _inline("A **bold** claim [3] with #hash.")
    assert "#strong[bold]" in converted
    assert "\\[3\\]" in converted
    assert "\\#hash" in converted


def test_markdown_conversion_headings_lists_and_guards() -> None:
    markdown = "### Subhead\n\n- item one [1]\n1. numbered\n= not a heading\nplain line"
    converted = _markdown_to_typst(markdown)
    assert "=== Subhead" in converted
    assert "- item one \\[1\\]" in converted
    assert "+ numbered" in converted
    assert "\\= not a heading" in converted
    assert "plain line" in converted


def test_render_pdf_compiles_offline(tmp_path: Path) -> None:
    draft = DraftReport(
        topic="Sand batteries, locally",
        sections=[
            ReportSection(
                title="Overview",
                markdown="A claim [1]. **Bold** finding [2].\n\n### Detail\n\n- point [1]",
            ),
            ReportSection(title="Conclusions", markdown="Wrap up [2]."),
        ],
    )
    chart = tmp_path / "chart.svg"
    chart.write_text(TINY_SVG, encoding="utf-8")
    out = render_pdf(
        draft,
        make_citations(),
        Ledger().summary(),
        make_manifest(),
        {"funnel": chart},
        tmp_path / "out" / "report.pdf",
    )
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 5_000
