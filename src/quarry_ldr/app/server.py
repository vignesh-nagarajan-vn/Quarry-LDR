"""The local app's HTTP surface. Binds to loopback; keys never reach the page."""

from __future__ import annotations

import asyncio
import os
import subprocess
import webbrowser
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quarry_ldr.app.jobs import Job, JobStatus, JobStore, Runner, probe_search
from quarry_ldr.app.library import Library, Passage, RunDetail, RunSummary, valid_run_id
from quarry_ldr.config import QuarryConfig
from quarry_ldr.preflight import DOCKER_REMEDIATION, compose_command, run_preflight

_STATIC = Path(__file__).parent / "static"
_LOOPBACK = {"127.0.0.1", "localhost"}
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    # Revalidate, so a page from before an update cannot keep running old code.
    "Cache-Control": "no-cache",
}
_MAX_TOPIC_CHARS = 300
_COMPOSE_TIMEOUT_S = 120
_SLUG_WORDS = 8


class EngineOption(BaseModel):
    id: Literal["local", "assisted", "premium"]
    name: str
    summary: str
    measured_cost: str
    default_cap: float
    needs_key: bool


# Measured before Update 1; DECISIONS.md holds the source numbers.
ENGINES = [
    EngineOption(
        id="local",
        name="Local",
        summary="Your GPU does all the reading and writing.",
        measured_cost="$0",
        default_cap=0.0,
        needs_key=False,
    ),
    EngineOption(
        id="assisted",
        name="Assisted",
        summary="A local draft, with Haiku checking gaps and polishing.",
        measured_cost="$0.02 to $0.12 measured",
        default_cap=0.5,
        needs_key=True,
    ),
    EngineOption(
        id="premium",
        name="Premium",
        summary="Claude plans, audits, and writes over your local evidence.",
        measured_cost="$1.36 to $2.88 measured",
        default_cap=4.5,
        needs_key=True,
    ),
]
_ENGINES_BY_ID = {engine.id: engine for engine in ENGINES}


class NewJob(BaseModel):
    topic: str = Field(min_length=3, max_length=_MAX_TOPIC_CHARS)
    engine: Literal["local", "assisted", "premium"] = "local"
    iterations: int = Field(default=3, ge=1, le=3)
    max_cost: float = Field(default=0.0, ge=0.0)
    confirm_spend: bool = False


class Options(BaseModel):
    engines: list[EngineOption]
    key_present: bool
    max_cost_allowed: float


class Check(BaseModel):
    name: str
    status: str
    detail: str


class Setup(BaseModel):
    checks: list[Check]
    ready: bool


class SearchHealth(BaseModel):
    healthy: bool


class ActionResult(BaseModel):
    ok: bool
    detail: str


class Report(BaseModel):
    html: str


def _slug(topic: str) -> str:
    words = "".join(c.lower() if c.isalnum() else " " for c in topic).split()
    return "-".join(words[:_SLUG_WORDS]) or "report"


def _pdf_name(detail: RunDetail) -> str:
    return f"{_slug(detail.topic)}-{detail.created_at[:10]}.pdf"


def _key_present(cfg: QuarryConfig) -> bool:
    return bool(cfg.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _run_id_or_404(run_id: str) -> str:
    if not valid_run_id(run_id):
        raise HTTPException(status_code=404, detail="Unknown run")
    return run_id


def _start_search() -> ActionResult:
    command = compose_command(["up", "-d"])
    if command is None:
        return ActionResult(ok=False, detail=DOCKER_REMEDIATION)
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=_COMPOSE_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ActionResult(ok=False, detail=f"Docker did not respond: {exc}")
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        return ActionResult(ok=False, detail=lines[-1] if lines else "docker compose failed")
    return ActionResult(ok=True, detail="Search is starting.")


def create_app(
    cfg: QuarryConfig,
    config_path: Path | None = None,
    open_url: str | None = None,
    runner: Runner | None = None,
) -> FastAPI:
    """``runner`` is injectable so tests drive the queue one tick at a time."""
    library = Library(cfg.run.data_dir)
    store = runner.store if runner is not None else JobStore(cfg.run.data_dir / "app.db")
    active = runner or Runner(cfg, store, library, config_path=config_path)
    allowed_hosts = _LOOPBACK

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await store.open()
        await active.recover()
        task = asyncio.create_task(active.run_forever())
        if open_url is not None:
            webbrowser.open(open_url)
        try:
            yield
        finally:
            task.cancel()
            active.stop_active()
            await store.close()

    app = FastAPI(title="Quarry", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Refuses DNS rebinding and other sites' pages: this server can start paid runs.
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in allowed_hosts:
            return JSONResponse({"detail": "Forbidden host"}, status_code=403)
        origin = request.headers.get("origin")
        cross_site = origin is not None and urlsplit(origin).netloc != request.headers.get("host")
        if request.method not in ("GET", "HEAD") and cross_site:
            return JSONResponse({"detail": "Forbidden origin"}, status_code=403)
        response = await call_next(request)
        response.headers.update(_SECURITY_HEADERS)
        return response

    @app.get("/api/options")
    async def options() -> Options:
        return Options(
            engines=ENGINES, key_present=_key_present(cfg), max_cost_allowed=cfg.run.cost_cap_usd
        )

    @app.get("/api/setup")
    async def setup() -> Setup:
        checks = await asyncio.to_thread(run_preflight, cfg)
        return Setup(
            checks=[Check(name=c.name, status=c.status, detail=c.detail) for c in checks],
            ready=all(c.status != "missing" for c in checks),
        )

    @app.post("/api/setup/search")
    async def start_search() -> ActionResult:
        return await asyncio.to_thread(_start_search)

    @app.get("/api/search-health")
    async def search_health() -> SearchHealth:
        return SearchHealth(healthy=await probe_search(cfg, "offshore wind project capacity"))

    @app.get("/api/jobs")
    async def jobs() -> list[Job]:
        return await store.all()

    @app.post("/api/jobs", status_code=201)
    async def add_job(body: NewJob) -> Job:
        engine = _ENGINES_BY_ID[body.engine]
        if engine.needs_key:
            if not _key_present(cfg):
                raise HTTPException(status_code=400, detail="This engine needs an API key.")
            if not body.confirm_spend or body.max_cost <= 0:
                raise HTTPException(status_code=400, detail="Confirm a spending limit first.")
            if body.max_cost > cfg.run.cost_cap_usd:
                limit = cfg.run.cost_cap_usd
                raise HTTPException(status_code=400, detail=f"The limit per run is ${limit:.2f}.")
        job = active.new_job(body.topic.strip(), body.engine, body.max_cost, body.iterations)
        return await active.add(job)

    @app.post("/api/jobs/{job_id}/stop")
    async def stop_job(job_id: str) -> Job:
        job = await active.stop(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return job

    @app.post("/api/jobs/{job_id}/resume", status_code=201)
    async def resume_job(job_id: str) -> Job:
        job = await store.get(job_id)
        if job is None or job.run_id is None:
            raise HTTPException(status_code=404, detail="Nothing to resume")
        if job.status not in (JobStatus.STOPPED, JobStatus.FAILED):
            raise HTTPException(status_code=409, detail="This job is still active.")
        return await active.add(
            active.new_job(job.topic, job.engine, job.max_cost, job.iterations, job.run_id)
        )

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def forget_job(job_id: str) -> None:
        job = await store.get(job_id)
        if job is not None and job.status in (
            JobStatus.RUNNING,
            JobStatus.QUEUED,
            JobStatus.WAITING_SEARCH,
        ):
            raise HTTPException(status_code=409, detail="Stop this job first.")
        await store.delete(job_id)

    @app.get("/api/runs")
    async def runs(q: str = "") -> list[RunSummary]:
        return await asyncio.to_thread(library.list_runs, q)

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> RunDetail:
        detail = await asyncio.to_thread(library.detail, _run_id_or_404(run_id))
        if detail is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        return detail

    @app.get("/api/runs/{run_id}/report")
    async def report(run_id: str) -> Report:
        html = await asyncio.to_thread(library.report_html, _run_id_or_404(run_id))
        if html is None:
            raise HTTPException(status_code=404, detail="This run has no report yet.")
        return Report(html=html)

    @app.get("/api/runs/{run_id}/passages/{number}")
    async def passage(run_id: str, number: int) -> Passage:
        found = await asyncio.to_thread(library.passage, _run_id_or_404(run_id), number)
        if found is None:
            raise HTTPException(status_code=404, detail="Unknown citation")
        return found

    @app.get("/api/runs/{run_id}/pdf")
    async def pdf(run_id: str) -> FileResponse:
        path = await asyncio.to_thread(library.pdf_path, _run_id_or_404(run_id))
        detail = await asyncio.to_thread(library.detail, run_id)
        if path is None or detail is None:
            raise HTTPException(status_code=404, detail="This run has no PDF.")
        return FileResponse(path, media_type="application/pdf", filename=_pdf_name(detail))

    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
    return app
