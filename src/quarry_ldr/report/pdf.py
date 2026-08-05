"""Branded PDF deliverable: per-run Typst source, compiled by the bundled
typst wheel, fully offline.

template.typ (checked in beside this module) owns the visual identity; this
module converts already-validated report content into Typst markup, escaping
every piece of model- or corpus-derived text, and compiles inside a temp
root so the sandboxed compiler sees only what this run produced.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import typst

from quarry_ldr.ledger import LedgerSummary
from quarry_ldr.pipeline.synthesize import DraftReport
from quarry_ldr.report.citations import CitationIndex
from quarry_ldr.report.render import RunManifest

_TEMPLATE_PATH = Path(__file__).parent / "template.typ"
# Backslash must come first so escapes are not themselves re-escaped.
_ESCAPE_CHARS = "\\#$[]*_`<>@~"
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_HEADING = re.compile(r"^(#{3,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_LINE_LEAD_MARKUP = re.compile(r"^(\s*)([=/+])")
_BOLD_OPEN = "\x00B\x00"
_BOLD_CLOSE = "\x00b\x00"


def _escape_typst(text: str) -> str:
    out = text
    for char in _ESCAPE_CHARS:
        out = out.replace(char, "\\" + char)
    return out


def _quoted(text: str) -> str:
    """Escaping for Typst string literals (template arguments)."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _inline(text: str) -> str:
    """Escape one line of prose, preserving **bold** as Typst strong."""
    protected = _BOLD.sub(lambda m: _BOLD_OPEN + m.group(1) + _BOLD_CLOSE, text)
    escaped = _escape_typst(protected)
    return escaped.replace(_BOLD_OPEN, "#strong[").replace(_BOLD_CLOSE, "]")


def _markdown_to_typst(markdown: str) -> str:
    """The subset SECTION_PROMPT allows: paragraphs, ### subsections, lists,
    **bold**, and [n] markers (which stay literal, matching the markdown)."""
    lines_out: list[str] = []
    for raw_line in markdown.splitlines():
        heading = _HEADING.match(raw_line)
        bullet = _BULLET.match(raw_line)
        numbered = _NUMBERED.match(raw_line)
        if heading:
            lines_out.append("=" * min(len(heading.group(1)), 5) + " " + _inline(heading.group(2)))
        elif bullet:
            lines_out.append("- " + _inline(bullet.group(1)))
        elif numbered:
            lines_out.append("+ " + _inline(numbered.group(1)))
        else:
            converted = _inline(raw_line)
            # A leading =, /, or + would read as Typst markup; neutralize it.
            lines_out.append(_LINE_LEAD_MARKUP.sub(r"\1\\\2", converted))
    return "\n".join(lines_out)


def _link(url: str) -> str:
    return f'#link("{_quoted(url)}")[{_escape_typst(url)}]'


def _build_source(
    draft: DraftReport,
    citations: CitationIndex,
    summary: LedgerSummary,
    manifest: RunManifest,
    chart_files: dict[str, str],
) -> str:
    parts: list[str] = [
        '#import "template.typ": quarry-report',
        "#show: quarry-report.with(",
        f'  title: "{_quoted(draft.topic)}",',
        f'  run-id: "{_quoted(manifest.run_id)}",',
        f'  date: "{manifest.finished_at:%Y-%m-%d}",',
        f'  engine: "{_quoted(manifest.models.get("engine", ""))}",',
        f'  cost: "USD {summary.total_cost_usd:.4f}",',
        ")",
        "",
    ]
    for section in draft.sections:
        parts.append("== " + _inline(section.title))
        parts.append("")
        parts.append(_markdown_to_typst(section.markdown))
        parts.append("")

    if chart_files:
        parts.append("== Run Charts")
        parts.append("")
        for filename in chart_files.values():
            parts.append(f'#figure(image("{_quoted(filename)}", width: 92%))')
            parts.append("")

    parts.append("== References")
    parts.append("")
    for citation in citations.all():
        title = _inline(citation.title or citation.url)
        anchor = f"chunk {citation.chunk_id}, chars {citation.start_char}-{citation.end_char}"
        parts.append(
            f"\\[{citation.number}\\] {title} - {_link(citation.url)} ({_escape_typst(anchor)})"
        )
        parts.append("")

    parts.append("== Run Facts")
    parts.append("")
    facts: list[tuple[str, str]] = [
        ("Run id", manifest.run_id),
        ("Engine", manifest.models.get("engine", "")),
        ("Iterations", str(manifest.iterations)),
        ("Sources fetched", str(manifest.n_urls_fetched)),
        ("Chunks", str(manifest.n_chunks)),
        ("After dedup", str(manifest.n_chunks_after_dedup)),
        ("Evidence chunks", str(manifest.n_chunks_evidence)),
        ("Claims checked", str(manifest.n_claims_checked)),
        (
            "Claims rewritten / dropped",
            f"{manifest.n_claims_rewritten} / {manifest.n_claims_dropped}",
        ),
        ("API spend", f"USD {summary.total_cost_usd:.4f}"),
    ]
    for stage, cost in sorted(summary.by_stage.items(), key=lambda kv: -kv[1]):
        if cost > 0:
            facts.append((f"Spend: {stage}", f"USD {cost:.4f}"))
    for key, value in facts:
        parts.append(f"- #strong[{_escape_typst(key)}:] {_escape_typst(value)}")
    parts.append("")
    return "\n".join(parts)


def render_pdf(
    draft: DraftReport,
    citations: CitationIndex,
    summary: LedgerSummary,
    manifest: RunManifest,
    chart_paths: dict[str, Path],
    out_path: Path,
) -> Path:
    """Compile the branded PDF next to the markdown report and return its path."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copy2(_TEMPLATE_PATH, root / "template.typ")
        chart_files: dict[str, str] = {}
        for name, path in chart_paths.items():
            target = root / f"chart-{name}.svg"
            shutil.copy2(path, target)
            chart_files[name] = target.name
        main = root / "main.typ"
        main.write_text(
            _build_source(draft, citations, summary, manifest, chart_files),
            encoding="utf-8",
            newline="\n",
        )
        pdf_bytes = typst.compile(str(main), root=str(root))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)
    return out_path
