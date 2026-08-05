"""SQLite-backed run store: every stage transition is a row, runs are resumable.

Schema (created on open):
    runs(run_id TEXT PK, topic, status, created_at, updated_at, iteration, config_snapshot JSON)
    stages(run_id, stage, iteration, status, started_at, finished_at, payload JSON, error,
           PRIMARY KEY (run_id, stage, iteration))
    ledger(seq INTEGER PK AUTOINCREMENT, run_id, entry JSON)

WAL journal mode keeps a crash mid-write from corrupting earlier rows; large
artifacts (fetched bodies, vectors) live on disk and in LanceDB, so stage
payloads stay small JSON.
"""

from __future__ import annotations

import enum
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import aiosqlite
from pydantic import BaseModel, Field

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 0,
    config_snapshot TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS stages (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    stage TEXT NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    PRIMARY KEY (run_id, stage, iteration)
);
CREATE TABLE IF NOT EXISTS ledger (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    entry TEXT NOT NULL
);
"""


class RunStatus(enum.StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Stage(enum.StrEnum):
    """Pipeline stages in execution order. Loop iterations reuse SEARCH..GAP."""

    PLAN = "plan"
    SEARCH = "search"
    FETCH = "fetch"
    EXTRACT = "extract"
    CHUNK = "chunk"
    EMBED = "embed"
    DEDUP = "dedup"
    INDEX = "index"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    TRIAGE = "triage"
    GAP = "gap"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"
    RENDER = "render"


class StageStatus(enum.StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunRecord(BaseModel):
    run_id: str
    topic: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    iteration: int = 0
    config_snapshot: dict[str, Any] = Field(default_factory=dict)


class StageRecord(BaseModel):
    run_id: str
    stage: Stage
    iteration: int
    status: StageStatus
    started_at: datetime
    finished_at: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def new_run_id() -> str:
    """Short, filesystem-safe run identifier."""
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_from_row(row: aiosqlite.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        topic=row["topic"],
        status=RunStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        iteration=row["iteration"],
        config_snapshot=json.loads(row["config_snapshot"]),
    )


def _stage_from_row(row: aiosqlite.Row) -> StageRecord:
    return StageRecord(
        run_id=row["run_id"],
        stage=Stage(row["stage"]),
        iteration=row["iteration"],
        status=StageStatus(row["status"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=(datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None),
        payload=json.loads(row["payload"]),
        error=row["error"],
    )


class RunStore:
    """Async SQLite (WAL) store for runs, stage transitions, and ledger entries.

    Usage::

        async with RunStore(path) as store:
            run = await store.create_run("topic", cfg.snapshot())
            await store.start_stage(run.run_id, Stage.PLAN)
            await store.complete_stage(run.run_id, Stage.PLAN, payload={...})
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("RunStore is not open; use `async with RunStore(...)`")
        return self._db

    async def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def create_run(self, topic: str, config_snapshot: dict[str, Any]) -> RunRecord:
        now = _now()
        run_id = new_run_id()
        await self.db.execute(
            "INSERT INTO runs (run_id, topic, status, created_at, updated_at, iteration,"
            " config_snapshot) VALUES (?, ?, ?, ?, ?, 0, ?)",
            (run_id, topic, RunStatus.CREATED, now, now, json.dumps(config_snapshot)),
        )
        await self.db.commit()
        return await self.get_run(run_id)

    async def get_run(self, run_id: str) -> RunRecord:
        """Raise KeyError if the run does not exist."""
        cursor = await self.db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        return _run_from_row(row)

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        cursor = await self.db.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, run_id LIMIT ?", (limit,)
        )
        return [_run_from_row(row) for row in await cursor.fetchall()]

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        await self._touch_run(run_id, "status = ?", (status,))

    async def set_run_iteration(self, run_id: str, iteration: int) -> None:
        await self._touch_run(run_id, "iteration = ?", (iteration,))

    async def _touch_run(self, run_id: str, set_clause: str, params: tuple[Any, ...]) -> None:
        cursor = await self.db.execute(
            f"UPDATE runs SET {set_clause}, updated_at = ? WHERE run_id = ?",
            (*params, _now(), run_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"unknown run_id: {run_id}")
        await self.db.commit()

    async def start_stage(self, run_id: str, stage: Stage, iteration: int = 0) -> StageRecord:
        """Insert (or reset, on retry after failure) a stage row as RUNNING."""
        await self.get_run(run_id)  # KeyError on unknown run
        now = _now()
        await self.db.execute(
            "INSERT INTO stages (run_id, stage, iteration, status, started_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id, stage, iteration) DO UPDATE SET"
            " status = excluded.status, started_at = excluded.started_at,"
            " finished_at = NULL, payload = '{}', error = NULL",
            (run_id, stage, iteration, StageStatus.RUNNING, now),
        )
        await self.db.commit()
        record = await self.get_stage(run_id, stage, iteration)
        assert record is not None
        return record

    async def complete_stage(
        self, run_id: str, stage: Stage, payload: dict[str, Any], iteration: int = 0
    ) -> StageRecord:
        return await self._finish_stage(
            run_id, stage, iteration, StageStatus.COMPLETED, payload=payload, error=None
        )

    async def fail_stage(
        self, run_id: str, stage: Stage, error: str, iteration: int = 0
    ) -> StageRecord:
        return await self._finish_stage(
            run_id, stage, iteration, StageStatus.FAILED, payload=None, error=error
        )

    async def _finish_stage(
        self,
        run_id: str,
        stage: Stage,
        iteration: int,
        status: StageStatus,
        payload: dict[str, Any] | None,
        error: str | None,
    ) -> StageRecord:
        sets = ["status = ?", "finished_at = ?"]
        params: list[Any] = [status, _now()]
        if payload is not None:
            sets.append("payload = ?")
            params.append(json.dumps(payload))
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        cursor = await self.db.execute(
            f"UPDATE stages SET {', '.join(sets)} WHERE run_id = ? AND stage = ? AND iteration = ?",
            (*params, run_id, stage, iteration),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"stage {stage} (iteration {iteration}) was never started for {run_id}")
        await self.db.commit()
        record = await self.get_stage(run_id, stage, iteration)
        assert record is not None
        return record

    async def get_stage(self, run_id: str, stage: Stage, iteration: int = 0) -> StageRecord | None:
        cursor = await self.db.execute(
            "SELECT * FROM stages WHERE run_id = ? AND stage = ? AND iteration = ?",
            (run_id, stage, iteration),
        )
        row = await cursor.fetchone()
        return None if row is None else _stage_from_row(row)

    async def stages(self, run_id: str) -> list[StageRecord]:
        """All stage records for a run, in insertion order."""
        cursor = await self.db.execute(
            "SELECT * FROM stages WHERE run_id = ? ORDER BY rowid", (run_id,)
        )
        return [_stage_from_row(row) for row in await cursor.fetchall()]

    async def latest_completed_stage(self, run_id: str) -> StageRecord | None:
        """The most recently completed stage, for resume."""
        cursor = await self.db.execute(
            "SELECT * FROM stages WHERE run_id = ? AND status = ? ORDER BY rowid DESC LIMIT 1",
            (run_id, StageStatus.COMPLETED),
        )
        row = await cursor.fetchone()
        return None if row is None else _stage_from_row(row)

    async def append_ledger_entry(self, run_id: str, entry: dict[str, Any]) -> None:
        await self.db.execute(
            "INSERT INTO ledger (run_id, entry) VALUES (?, ?)", (run_id, json.dumps(entry))
        )
        await self.db.commit()

    async def ledger_entries(self, run_id: str) -> list[dict[str, Any]]:
        cursor = await self.db.execute(
            "SELECT entry FROM ledger WHERE run_id = ? ORDER BY seq", (run_id,)
        )
        return [json.loads(row["entry"]) for row in await cursor.fetchall()]
