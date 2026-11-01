"""Read-only views of past runs: library, progress, report text, and cited passages.

Everything comes from the run store and the report files, so a report started in
the terminal shows up here the same as one started in the app.
"""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from pydantic import BaseModel

_RUN_ID = re.compile(r"^[0-9a-f]{8,32}$")
_REFERENCE = re.compile(
    r"^\[(?P<number>\d+)\] (?P<title>.*?) - (?P<url>\S+) \(.*?chunk (?P<chunk>[0-9a-f]+), "
    r"chars \d+-\d+\)\s*$",
    re.MULTILINE,
)
_TAG = re.compile(r"(<[^>]+>)")
_MARKER = re.compile(r"\[(\d+)\]")
_SKIP_MARKERS_IN = {"a", "code", "pre"}
_MAX_REPORT_SEARCHES = 200

_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")


class RunSummary(BaseModel):
    run_id: str
    topic: str
    status: str
    created_at: str
    updated_at: str
    cost_usd: float
    has_report: bool
    has_pdf: bool


class StageView(BaseModel):
    stage: str
    iteration: int
    status: str
    started_at: str
    finished_at: str | None
    error: str | None


class RunDetail(RunSummary):
    stages: list[StageView]
    n_urls: int


class Passage(BaseModel):
    number: int
    title: str
    url: str
    text: str


def valid_run_id(run_id: str) -> bool:
    """Run ids become file names, so anything else is refused."""
    return _RUN_ID.match(run_id) is not None


def render_report_html(markdown: str) -> str:
    """Report markdown to HTML. Raw HTML is escaped, because a report quotes web
    pages, and each [n] marker becomes a button that opens its source passage."""
    html: str = _markdown.render(markdown)
    parts: list[str] = []
    skipping: list[str] = []
    for part in _TAG.split(html):
        if part.startswith("<"):
            name = part.strip("</>").split(" ")[0]
            if name in _SKIP_MARKERS_IN:
                if part.startswith("</"):
                    if skipping:
                        skipping.pop()
                else:
                    skipping.append(name)
            parts.append(part)
        elif skipping:
            parts.append(part)
        else:
            parts.append(
                _MARKER.sub(
                    lambda m: (
                        f'<button type="button" class="cite" data-n="{m[1]}">[{m[1]}]</button>'
                    ),
                    part,
                )
            )
    return "".join(parts)


@lru_cache(maxsize=8)
def _evidence_text(db_path: str, run_id: str, updated_at: str) -> dict[str, str]:
    """Chunk id to text for one run. Every cited chunk is evidence, and the far
    larger CHUNK payloads never need loading. ``updated_at`` keys the cache."""
    db = _connect(Path(db_path))
    try:
        rows = db.execute(
            "SELECT payload FROM stages WHERE run_id=? AND stage='triage' ORDER BY iteration",
            (run_id,),
        ).fetchall()
    finally:
        db.close()
    texts: dict[str, str] = {}
    for row in rows:
        for item in json.loads(row["payload"]).get("evidence", []):
            texts[item["chunk"]["chunk_id"]] = item["chunk"]["text"]
    return texts


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


class Library:
    """Queries over runs.db and data/reports. Sync: the server calls it in a thread."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "runs.db"

    def _report(self, run_id: str, suffix: str) -> Path:
        return self.data_dir / "reports" / f"report-{run_id}{suffix}"

    def _summary(self, row: sqlite3.Row, cost: float) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "topic": row["topic"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "cost_usd": cost,
            "has_report": self._report(row["run_id"], ".md").is_file(),
            "has_pdf": self._report(row["run_id"], ".pdf").is_file(),
        }

    def list_runs(self, query: str = "", limit: int = 100) -> list[RunSummary]:
        """Newest first. A query matches the topic or the report's own text."""
        if not self.db_path.is_file():
            return []
        needle = query.strip().lower()
        db = _connect(self.db_path)
        try:
            rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            costs = {
                row["run_id"]: row["cost"] or 0.0
                for row in db.execute(
                    "SELECT run_id, SUM(json_extract(entry, '$.cost_usd')) AS cost "
                    "FROM ledger GROUP BY run_id"
                )
            }
        finally:
            db.close()
        found: list[RunSummary] = []
        searched = 0
        for row in rows:
            if needle and needle not in row["topic"].lower():
                searched += 1
                if searched > _MAX_REPORT_SEARCHES or not self._report_mentions(row, needle):
                    continue
            found.append(RunSummary(**self._summary(row, costs.get(row["run_id"], 0.0))))
            if len(found) >= limit:
                break
        return found

    def _report_mentions(self, row: sqlite3.Row, needle: str) -> bool:
        path = self._report(row["run_id"], ".md")
        return path.is_file() and needle in path.read_text(encoding="utf-8").lower()

    def detail(self, run_id: str) -> RunDetail | None:
        if not self.db_path.is_file():
            return None
        db = _connect(self.db_path)
        try:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            stages = db.execute(
                "SELECT stage, iteration, status, started_at, finished_at, error FROM stages "
                "WHERE run_id=? ORDER BY started_at",
                (run_id,),
            ).fetchall()
            cost = db.execute(
                "SELECT SUM(json_extract(entry, '$.cost_usd')) AS cost FROM ledger WHERE run_id=?",
                (run_id,),
            ).fetchone()["cost"]
            urls = db.execute(
                "SELECT MAX(json_array_length(payload, '$.urls')) AS n FROM stages "
                "WHERE run_id=? AND stage='search' AND status='completed'",
                (run_id,),
            ).fetchone()["n"]
        finally:
            db.close()
        return RunDetail(
            **self._summary(row, cost or 0.0),
            stages=[StageView(**dict(stage)) for stage in stages],
            n_urls=urls or 0,
        )

    def find_run(self, topic: str, since: str) -> str | None:
        """The run a just-started process created. The CLI prints the id only at the end."""
        if not self.db_path.is_file():
            return None
        db = _connect(self.db_path)
        try:
            row = db.execute(
                "SELECT run_id FROM runs WHERE topic=? AND created_at>=? ORDER BY created_at",
                (topic, since),
            ).fetchone()
        finally:
            db.close()
        return None if row is None else str(row["run_id"])

    def first_search_empty(self, run_id: str) -> bool:
        """The first search finished with no URLs: engines are refusing, so stop."""
        db = _connect(self.db_path)
        try:
            row = db.execute(
                "SELECT json_array_length(payload, '$.urls') AS n FROM stages "
                "WHERE run_id=? AND stage='search' AND iteration=0 AND status='completed'",
                (run_id,),
            ).fetchone()
        finally:
            db.close()
        return row is not None and row["n"] == 0

    def report_html(self, run_id: str) -> str | None:
        path = self._report(run_id, ".md")
        return render_report_html(path.read_text(encoding="utf-8")) if path.is_file() else None

    def pdf_path(self, run_id: str) -> Path | None:
        path = self._report(run_id, ".pdf")
        return path if path.is_file() else None

    def passage(self, run_id: str, number: int) -> Passage | None:
        """The exact chunk behind one [n], from the report's own reference list."""
        report = self._report(run_id, ".md")
        detail = self.detail(run_id)
        if detail is None or not report.is_file():
            return None
        for match in _REFERENCE.finditer(report.read_text(encoding="utf-8")):
            if int(match["number"]) == number:
                texts = _evidence_text(str(self.db_path), run_id, detail.updated_at)
                url = match["url"] if match["url"].startswith(("http://", "https://")) else ""
                return Passage(
                    number=number,
                    title=match["title"],
                    url=url,
                    text=texts.get(match["chunk"], ""),
                )
        return None
