"""SQLite-backed run store: every stage transition is a row, runs are resumable.

Schema (created on open):
    runs(run_id TEXT PK, topic, status, created_at, updated_at, iteration, config_snapshot JSON)
    stages(run_id, stage, iteration, status, started_at, finished_at, payload JSON, error,
           PRIMARY KEY (run_id, stage, iteration))
    ledger(run_id, seq, entry JSON)

Implemented in M1.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pydantic import BaseModel, Field


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

    async def open(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

    async def get_run(self, run_id: str) -> RunRecord:
        """Raise KeyError if the run does not exist."""
        raise NotImplementedError

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        raise NotImplementedError

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        raise NotImplementedError

    async def set_run_iteration(self, run_id: str, iteration: int) -> None:
        raise NotImplementedError

    async def start_stage(self, run_id: str, stage: Stage, iteration: int = 0) -> StageRecord:
        raise NotImplementedError

    async def complete_stage(
        self, run_id: str, stage: Stage, payload: dict[str, Any], iteration: int = 0
    ) -> StageRecord:
        raise NotImplementedError

    async def fail_stage(
        self, run_id: str, stage: Stage, error: str, iteration: int = 0
    ) -> StageRecord:
        raise NotImplementedError

    async def get_stage(self, run_id: str, stage: Stage, iteration: int = 0) -> StageRecord | None:
        raise NotImplementedError

    async def stages(self, run_id: str) -> list[StageRecord]:
        """All stage records for a run, in insertion order."""
        raise NotImplementedError

    async def latest_completed_stage(self, run_id: str) -> StageRecord | None:
        """The most recently completed stage, for resume."""
        raise NotImplementedError

    async def append_ledger_entry(self, run_id: str, entry: dict[str, Any]) -> None:
        raise NotImplementedError

    async def ledger_entries(self, run_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError
