"""The report queue: one run at a time, spaced out, and only while search answers.

Each job is a ``quarry research`` or ``quarry resume`` process, so the app runs the
same engine as the terminal and a crash in a run cannot take the app down. Only one
process runs at a time, which keeps the GPU arbiter's budget true: the arbiter
guards one process.
"""

from __future__ import annotations

import asyncio
import enum
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Literal, Protocol

import aiosqlite
import httpx
from pydantic import BaseModel

from quarry_ldr.app.library import Library
from quarry_ldr.config import QuarryConfig
from quarry_ldr.logging import get_logger

logger = get_logger(component="app_jobs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    doc TEXT NOT NULL
);
"""
_POLL_SECONDS = 2.0
_LOG_TAIL_BYTES = 2000
_PROBE_QUERIES = (
    "tidal power projects capacity",
    "geothermal heat pump installation cost",
    "battery recycling plant output",
)
_MIN_PROBE_RESULTS = 10


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    WAITING_SEARCH = "waiting_search"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


_OPEN = (JobStatus.QUEUED, JobStatus.WAITING_SEARCH)


class Job(BaseModel):
    id: str
    topic: str
    engine: Literal["local", "assisted", "premium"]
    max_cost: float
    iterations: int
    status: JobStatus = JobStatus.QUEUED
    run_id: str | None = None
    resume: bool = False
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    not_before: datetime | None = None
    healthy_probes: int = 0
    note: str = ""


class JobProcess(Protocol):
    def poll(self) -> int | None: ...

    def stop(self) -> None: ...


class _Process:
    """A run's process. Stopping takes the whole tree: a run owns a llama-server, and
    an orphan would keep holding the VRAM the next run needs."""

    def __init__(self, command: list[str], log: IO[bytes], env: dict[str, str]) -> None:
        self._popen = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, **env},
            start_new_session=sys.platform != "win32",
        )

    def poll(self) -> int | None:
        code: int | None = self._popen.poll()
        return code

    def stop(self) -> None:
        if self._popen.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(self._popen.pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        else:
            os.killpg(self._popen.pid, signal.SIGTERM)


class JobStore:
    """Jobs survive an app restart, in their own file beside runs.db."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("JobStore is not open")
        return self._db

    async def save(self, job: Job) -> None:
        await self._conn().execute(
            "INSERT INTO jobs (id, doc) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET doc=excluded.doc",
            (job.id, job.model_dump_json()),
        )
        await self._conn().commit()

    async def all(self) -> list[Job]:
        """Oldest first, which is also the order the queue runs in."""
        async with self._conn().execute("SELECT doc FROM jobs ORDER BY seq") as cursor:
            return [Job.model_validate_json(row[0]) async for row in cursor]

    async def get(self, job_id: str) -> Job | None:
        async with self._conn().execute("SELECT doc FROM jobs WHERE id=?", (job_id,)) as cursor:
            row = await cursor.fetchone()
        return None if row is None else Job.model_validate_json(row[0])

    async def delete(self, job_id: str) -> None:
        await self._conn().execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await self._conn().commit()


async def probe_search(cfg: QuarryConfig, query: str) -> bool:
    """One free query: does a general engine answer with real results?"""
    try:
        async with httpx.AsyncClient(timeout=cfg.search.timeout_s) as client:
            response = await client.get(
                f"{cfg.search.searxng_url.rstrip('/')}/search",
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
            return len(response.json().get("results", [])) >= _MIN_PROBE_RESULTS
    except (httpx.HTTPError, ValueError):
        return False


def _run_env(job: Job) -> dict[str, str]:
    """A resumed run takes its engine and limit from the process, not its first start,
    so a paid run must not resume on another engine or under another limit."""
    return {
        "PYTHONUTF8": "1",
        "QUARRY_ENGINE__MODE": job.engine,
        "QUARRY_RUN__COST_CAP_USD": str(job.max_cost),
        "QUARRY_RUN__MAX_ITERATIONS": str(job.iterations),
    }


def _command(job: Job, config_path: Path | None) -> list[str]:
    command = [sys.executable, "-m", "quarry_ldr.cli"]
    if job.resume and job.run_id is not None:
        command += ["resume", job.run_id]
    else:
        command += [
            "research",
            job.topic,
            "--engine",
            job.engine,
            "--max-cost",
            str(job.max_cost),
            "--max-iterations",
            str(job.iterations),
        ]
    if config_path is not None:
        command += ["--config", str(config_path)]
    return command


Spawn = Callable[[Job, Path], JobProcess]
Probe = Callable[[str], Awaitable[bool]]


class Runner:
    """Moves the queue forward. ``step`` is one tick, so tests need no sleeping."""

    def __init__(
        self,
        cfg: QuarryConfig,
        store: JobStore,
        library: Library,
        spawn: Spawn | None = None,
        probe: Probe | None = None,
        now: Callable[[], datetime] | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.library = library
        self._config_path = config_path
        self._spawn = spawn or self._spawn_process
        self._probe = probe or (lambda query: probe_search(cfg, query))
        self._now = now or (lambda: datetime.now(UTC))
        self._active: tuple[str, JobProcess] | None = None
        self._rest_until: datetime | None = None
        self._n_probes = 0
        self._log_handles: dict[str, IO[bytes]] = {}

    def log_path(self, job_id: str) -> Path:
        return self.cfg.run.data_dir / "logs" / f"app-{job_id}.log"

    def _spawn_process(self, job: Job, log_path: Path) -> JobProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab")
        self._log_handles[job.id] = handle
        return _Process(_command(job, self._config_path), handle, _run_env(job))

    async def recover(self) -> None:
        """A job still marked running lost its app; its checkpoints make it resumable."""
        for job in await self.store.all():
            if job.status == JobStatus.RUNNING:
                job.status = JobStatus.STOPPED
                job.note = "The app closed during this run. Resume continues from its checkpoints."
                job.finished_at = self._now()
                await self.store.save(job)

    async def add(self, job: Job) -> Job:
        await self.store.save(job)
        return job

    def new_job(
        self,
        topic: str,
        engine: Literal["local", "assisted", "premium"],
        max_cost: float,
        iterations: int,
        run_id: str | None = None,
    ) -> Job:
        return Job(
            id=uuid.uuid4().hex[:12],
            topic=topic,
            engine=engine,
            max_cost=max_cost,
            iterations=iterations,
            run_id=run_id,
            resume=run_id is not None,
            created_at=self._now(),
        )

    async def stop(self, job_id: str) -> Job | None:
        """Stop a running job, or take a waiting one off the queue."""
        job = await self.store.get(job_id)
        if job is None:
            return None
        if self._active is not None and self._active[0] == job_id:
            self._active[1].stop()
            self._active = None
            self._close_log(job_id)
            self._rest_until = self._now() + timedelta(minutes=self.cfg.app.queue_spacing_minutes)
        if job.status in (*_OPEN, JobStatus.RUNNING):
            job.status = JobStatus.STOPPED
            job.finished_at = self._now()
            job.note = "Stopped. Resume continues from its checkpoints." if job.run_id else ""
            await self.store.save(job)
        return job

    def stop_active(self) -> None:
        if self._active is not None:
            self._active[1].stop()

    def _close_log(self, job_id: str) -> None:
        handle = self._log_handles.pop(job_id, None)
        if handle is not None:
            handle.close()

    def _log_tail(self, job_id: str) -> str:
        path = self.log_path(job_id)
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - _LOG_TAIL_BYTES))
            text = handle.read().decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1][:400] if lines else ""

    async def _finish(self, job: Job, status: JobStatus, note: str) -> None:
        job.status = status
        job.note = note
        job.finished_at = self._now()
        await self.store.save(job)
        self._active = None
        self._close_log(job.id)
        self._rest_until = self._now() + timedelta(minutes=self.cfg.app.queue_spacing_minutes)

    async def _watch_active(self, job_id: str, process: JobProcess) -> None:
        job = await self.store.get(job_id)
        if job is None:
            process.stop()
            self._active = None
            return
        if job.run_id is None and job.started_at is not None:
            job.run_id = await asyncio.to_thread(
                self.library.find_run, job.topic, job.started_at.isoformat()
            )
            if job.run_id is not None:
                await self.store.save(job)
        if job.run_id is not None and await asyncio.to_thread(
            self.library.first_search_empty, job.run_id
        ):
            # Stop before later stages run hollow and any paid call fires on nothing.
            process.stop()
            await self._finish(
                job, JobStatus.FAILED, "Search returned no results, so the run was stopped."
            )
            return
        code = process.poll()
        if code == 0:
            await self._finish(job, JobStatus.COMPLETED, "")
        elif code is not None:
            await self._finish(job, JobStatus.FAILED, self._log_tail(job.id))

    async def _advance(self, job: Job, now: datetime) -> None:
        self._n_probes += 1
        query = _PROBE_QUERIES[self._n_probes % len(_PROBE_QUERIES)]
        job.status = JobStatus.WAITING_SEARCH
        if not await self._probe(query):
            minutes = self.cfg.app.search_retry_minutes
            job.healthy_probes = 0
            job.not_before = now + timedelta(minutes=minutes)
            job.note = f"Search engines are refusing queries. Trying again in {minutes} minutes."
            await self.store.save(job)
            return
        job.healthy_probes += 1
        if job.healthy_probes < self.cfg.app.healthy_probes_required:
            job.not_before = now + timedelta(seconds=self.cfg.app.probe_spacing_seconds)
            job.note = "Checking that search is healthy."
            await self.store.save(job)
            return
        job.status = JobStatus.RUNNING
        job.started_at = now
        job.not_before = None
        job.note = ""
        await self.store.save(job)
        self._active = (job.id, self._spawn(job, self.log_path(job.id)))

    async def step(self) -> None:
        if self._active is not None:
            await self._watch_active(*self._active)
            return
        now = self._now()
        if self._rest_until is not None and now < self._rest_until:
            return
        waiting = [job for job in await self.store.all() if job.status in _OPEN]
        if not waiting:
            return
        job = waiting[0]
        if job.not_before is None or now >= job.not_before:
            await self._advance(job, now)

    async def run_forever(self) -> None:
        while True:
            try:
                await self.step()
            except (OSError, aiosqlite.Error) as exc:
                logger.warning("app_runner_step_failed", error=str(exc))
            await asyncio.sleep(_POLL_SECONDS)
