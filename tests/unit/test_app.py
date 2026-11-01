"""The local app: queue rules, library views, and the page's safety boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from quarry_ldr.app.jobs import Job, JobStatus, JobStore, Runner, _command
from quarry_ldr.app.library import Library, render_report_html, valid_run_id
from quarry_ldr.app.server import create_app
from quarry_ldr.config import QuarryConfig
from quarry_ldr.state import RunStore, Stage

RUN_ID = "abc123def456"
REPORT = """# Topic

## Key findings

- Cells reach 40 percent [1].

## References

[1] A source - https://a.example/doc (Heading; chunk feedbeef, chars 0-20)
[2] Bad link - javascript:alert(1) (chunk beefcafe, chars 0-5)
"""


class FakeProcess:
    def __init__(self) -> None:
        self.code: int | None = None
        self.stopped = False

    def poll(self) -> int | None:
        return self.code

    def stop(self) -> None:
        self.stopped = True


class Harness:
    """A runner with a controllable clock, search answer, and processes."""

    def __init__(self, cfg: QuarryConfig, store: JobStore) -> None:
        self.now = datetime(2026, 11, 1, tzinfo=UTC)
        self.healthy = True
        self.spawned: list[tuple[Job, FakeProcess]] = []

        async def probe(_: str) -> bool:
            return self.healthy

        def spawn(job: Job, _: Path) -> FakeProcess:
            process = FakeProcess()
            self.spawned.append((job, process))
            return process

        self.runner = Runner(
            cfg, store, Library(cfg.run.data_dir), spawn=spawn, probe=probe, now=lambda: self.now
        )

    def tick(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)

    async def add(self, topic: str = "A topic") -> Job:
        return await self.runner.add(self.runner.new_job(topic, "local", 0.0, 3))

    async def start_next(self) -> None:
        """Three healthy probes, spaced as the queue requires."""
        for _ in range(3):
            await self.runner.step()
            self.tick(self.runner.cfg.app.probe_spacing_seconds)


@pytest.fixture
def app_cfg(tmp_path: Path) -> QuarryConfig:
    cfg = QuarryConfig(_env_file=None)
    cfg.run.data_dir = tmp_path
    return cfg


@pytest.fixture
async def store(app_cfg: QuarryConfig) -> AsyncIterator[JobStore]:
    job_store = JobStore(app_cfg.run.data_dir / "app.db")
    await job_store.open()
    yield job_store
    await job_store.close()


@pytest.fixture
def harness(app_cfg: QuarryConfig, store: JobStore) -> Harness:
    return Harness(app_cfg, store)


@pytest.fixture
async def client(app_cfg: QuarryConfig, harness: Harness) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(app_cfg, runner=harness.runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8420") as http:
        yield http


async def _saved_run(cfg: QuarryConfig, urls: list[str] | None = None) -> str:
    async with RunStore(cfg.run.data_dir / "runs.db") as runs:
        run = await runs.create_run("Iron-air batteries", {})
        await runs.start_stage(run.run_id, Stage.SEARCH, 0)
        found = {"urls": urls if urls is not None else ["https://a"]}
        await runs.complete_stage(run.run_id, Stage.SEARCH, found, 0)
        await runs.start_stage(run.run_id, Stage.TRIAGE, 0)
        evidence = [{"chunk": {"chunk_id": "feedbeef", "text": "The exact passage."}}]
        await runs.complete_stage(run.run_id, Stage.TRIAGE, {"evidence": evidence}, 0)
        await runs.append_ledger_entry(run.run_id, {"cost_usd": 0.25})
    reports = cfg.run.data_dir / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"report-{run.run_id}.md").write_text(REPORT, encoding="utf-8")
    (reports / f"report-{run.run_id}.pdf").write_bytes(b"%PDF-1.7")
    return run.run_id


# ---- Queue


async def test_a_run_starts_only_after_three_healthy_probes(harness: Harness) -> None:
    await harness.add()
    await harness.runner.step()
    harness.tick(harness.runner.cfg.app.probe_spacing_seconds)
    await harness.runner.step()
    assert harness.spawned == []
    harness.tick(harness.runner.cfg.app.probe_spacing_seconds)
    await harness.runner.step()
    assert len(harness.spawned) == 1


async def test_probes_are_spaced_not_burst(harness: Harness) -> None:
    await harness.add()
    for _ in range(5):
        await harness.runner.step()
    assert harness.spawned == []


async def test_a_refused_probe_waits_and_starts_counting_again(harness: Harness) -> None:
    job = await harness.add()
    await harness.runner.step()
    harness.healthy = False
    harness.tick(harness.runner.cfg.app.probe_spacing_seconds)
    await harness.runner.step()
    saved = await harness.runner.store.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.WAITING_SEARCH
    assert saved.healthy_probes == 0
    assert "refusing" in saved.note
    harness.healthy = True
    await harness.runner.step()  # still inside the retry wait
    assert (await harness.runner.store.get(job.id)).healthy_probes == 0  # type: ignore[union-attr]
    harness.tick(harness.runner.cfg.app.search_retry_minutes * 60)
    await harness.start_next()
    assert len(harness.spawned) == 1


async def test_only_one_run_at_a_time_with_a_rest_between(harness: Harness) -> None:
    await harness.add("First")
    second = await harness.add("Second")
    await harness.start_next()
    for _ in range(6):
        await harness.runner.step()
        harness.tick(harness.runner.cfg.app.probe_spacing_seconds)
    assert len(harness.spawned) == 1

    harness.spawned[0][1].code = 0
    await harness.runner.step()
    await harness.start_next()
    assert len(harness.spawned) == 1  # resting

    harness.tick(harness.runner.cfg.app.queue_spacing_minutes * 60)
    await harness.start_next()
    assert [job.topic for job, _ in harness.spawned] == ["First", "Second"]
    assert (await harness.runner.store.get(second.id)).status == JobStatus.RUNNING  # type: ignore[union-attr]


async def test_a_failed_run_keeps_the_last_log_line(harness: Harness) -> None:
    job = await harness.add()
    await harness.start_next()
    log = harness.runner.log_path(job.id)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("starting\nmodel file is missing\n", encoding="utf-8")
    harness.spawned[0][1].code = 1
    await harness.runner.step()
    saved = await harness.runner.store.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.FAILED
    assert saved.note == "model file is missing"


async def test_an_empty_first_search_stops_the_run(harness: Harness, app_cfg: QuarryConfig) -> None:
    run_id = await _saved_run(app_cfg, urls=[])
    job = await harness.add()
    await harness.start_next()
    saved = await harness.runner.store.get(job.id)
    assert saved is not None
    saved.run_id = run_id
    await harness.runner.store.save(saved)
    await harness.runner.step()
    assert harness.spawned[0][1].stopped
    final = await harness.runner.store.get(job.id)
    assert final is not None
    assert final.status == JobStatus.FAILED
    assert "no results" in final.note


async def test_stopping_a_run_stops_its_process(harness: Harness) -> None:
    job = await harness.add()
    await harness.start_next()
    await harness.runner.stop(job.id)
    assert harness.spawned[0][1].stopped
    assert (await harness.runner.store.get(job.id)).status == JobStatus.STOPPED  # type: ignore[union-attr]


async def test_a_run_the_app_lost_is_marked_resumable(harness: Harness) -> None:
    job = await harness.add()
    job.status = JobStatus.RUNNING
    await harness.runner.store.save(job)
    await harness.runner.recover()
    saved = await harness.runner.store.get(job.id)
    assert saved is not None
    assert saved.status == JobStatus.STOPPED
    assert "Resume" in saved.note


def test_commands_match_the_terminal(harness: Harness) -> None:
    fresh = harness.runner.new_job("A topic", "premium", 4.5, 1)
    assert _command(fresh, None)[3:] == [
        "research",
        "A topic",
        "--engine",
        "premium",
        "--max-cost",
        "4.5",
        "--max-iterations",
        "1",
    ]
    again = harness.runner.new_job("A topic", "local", 0.0, 3, run_id=RUN_ID)
    assert _command(again, None)[3:] == ["resume", RUN_ID]


# ---- Library


async def test_library_lists_runs_with_their_cost(app_cfg: QuarryConfig) -> None:
    run_id = await _saved_run(app_cfg)
    runs = Library(app_cfg.run.data_dir).list_runs()
    assert [run.run_id for run in runs] == [run_id]
    assert runs[0].cost_usd == 0.25
    assert runs[0].has_report
    assert runs[0].has_pdf


async def test_library_search_reads_topics_and_report_text(app_cfg: QuarryConfig) -> None:
    await _saved_run(app_cfg)
    library = Library(app_cfg.run.data_dir)
    assert len(library.list_runs("iron-AIR")) == 1
    assert len(library.list_runs("40 percent")) == 1
    assert library.list_runs("nothing like this") == []


def test_library_is_empty_before_any_run(tmp_path: Path) -> None:
    library = Library(tmp_path)
    assert library.list_runs() == []
    assert library.detail(RUN_ID) is None
    assert library.find_run("t", "2026") is None


async def test_detail_shows_stages_and_sources(app_cfg: QuarryConfig) -> None:
    run_id = await _saved_run(app_cfg)
    detail = Library(app_cfg.run.data_dir).detail(run_id)
    assert detail is not None
    assert [stage.stage for stage in detail.stages] == ["search", "triage"]
    assert detail.n_urls == 1


async def test_a_citation_opens_its_exact_passage(app_cfg: QuarryConfig) -> None:
    run_id = await _saved_run(app_cfg)
    library = Library(app_cfg.run.data_dir)
    found = library.passage(run_id, 1)
    assert found is not None
    assert found.text == "The exact passage."
    assert found.url == "https://a.example/doc"
    assert library.passage(run_id, 99) is None


async def test_a_passage_never_returns_a_script_link(app_cfg: QuarryConfig) -> None:
    run_id = await _saved_run(app_cfg)
    found = Library(app_cfg.run.data_dir).passage(run_id, 2)
    assert found is not None
    assert found.url == ""


def test_report_html_escapes_what_web_pages_put_in_it() -> None:
    html = render_report_html(
        'A page said <script>alert(1)</script> and <img src=x onerror="alert(2)"> [1].\n\n'
        "[click](javascript:alert(3))"
    )
    assert "<script" not in html
    assert "<img" not in html
    assert 'href="javascript' not in html


def test_report_html_turns_markers_into_buttons_outside_links() -> None:
    html = render_report_html("A fact [3]. [see [4]](https://a.example) `[5]`")
    assert 'data-n="3"' in html
    assert 'data-n="4"' not in html
    assert 'data-n="5"' not in html


@pytest.mark.parametrize("bad", ["../../etc", "UPPERCASE1", "abc", "aaaaaaaa/../x"])
def test_run_ids_that_could_become_paths_are_refused(bad: str) -> None:
    assert not valid_run_id(bad)


# ---- Server boundaries


async def test_another_host_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/jobs", headers={"Host": "evil.example"})
    assert response.status_code == 403


async def test_another_site_cannot_post(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/jobs", json={"topic": "A topic"}, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403


async def test_the_app_itself_can_post(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/jobs", json={"topic": "A topic"}, headers={"Origin": "http://127.0.0.1:8420"}
    )
    assert response.status_code == 201
    assert response.json()["engine"] == "local"


async def test_every_response_carries_the_content_policy(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/options")
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-cache"


async def test_a_paid_run_needs_a_key(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, app_cfg: QuarryConfig
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app_cfg.anthropic_api_key = None
    response = await client.post(
        "/api/jobs",
        json={"topic": "A topic", "engine": "premium", "max_cost": 1.0, "confirm_spend": True},
    )
    assert response.status_code == 400
    assert "API key" in response.json()["detail"]


async def test_a_paid_run_needs_an_accepted_limit(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")
    unconfirmed = {"topic": "A topic", "engine": "assisted", "max_cost": 0.5}
    assert (await client.post("/api/jobs", json=unconfirmed)).status_code == 400
    no_limit = {"topic": "A topic", "engine": "assisted", "max_cost": 0, "confirm_spend": True}
    assert (await client.post("/api/jobs", json=no_limit)).status_code == 400
    too_high = {"topic": "A topic", "engine": "premium", "max_cost": 99, "confirm_spend": True}
    assert (await client.post("/api/jobs", json=too_high)).status_code == 400
    assert (await client.get("/api/jobs")).json() == []
    allowed = {"topic": "A topic", "engine": "assisted", "max_cost": 0.5, "confirm_spend": True}
    assert (await client.post("/api/jobs", json=allowed)).status_code == 201


async def test_options_say_whether_a_key_exists_not_what_it_is(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")
    response = await client.get("/api/options")
    assert response.json()["key_present"] is True
    assert "test-value" not in response.text


async def test_a_stopped_job_resumes_as_a_new_job(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    job = await harness.add()
    job.status = JobStatus.STOPPED
    job.run_id = RUN_ID
    await harness.runner.store.save(job)
    response = await client.post(f"/api/jobs/{job.id}/resume")
    assert response.status_code == 201
    assert response.json()["resume"] is True
    assert response.json()["run_id"] == RUN_ID


async def test_an_active_job_cannot_be_resumed_or_removed(
    client: httpx.AsyncClient, harness: Harness
) -> None:
    job = await harness.add()
    job.run_id = RUN_ID
    await harness.runner.store.save(job)
    assert (await client.post(f"/api/jobs/{job.id}/resume")).status_code == 409
    assert (await client.delete(f"/api/jobs/{job.id}")).status_code == 409


async def test_a_finished_job_can_be_removed(client: httpx.AsyncClient, harness: Harness) -> None:
    job = await harness.add()
    job.status = JobStatus.COMPLETED
    await harness.runner.store.save(job)
    assert (await client.delete(f"/api/jobs/{job.id}")).status_code == 204
    assert (await client.get("/api/jobs")).json() == []


async def test_reports_and_passages_are_served(
    client: httpx.AsyncClient, app_cfg: QuarryConfig
) -> None:
    run_id = await _saved_run(app_cfg)
    assert 'class="cite"' in (await client.get(f"/api/runs/{run_id}/report")).json()["html"]
    passage = await client.get(f"/api/runs/{run_id}/passages/1")
    assert passage.json()["text"] == "The exact passage."
    assert (await client.get(f"/api/runs/{run_id}")).json()["n_urls"] == 1


async def test_a_pdf_downloads_under_a_readable_name(
    client: httpx.AsyncClient, app_cfg: QuarryConfig
) -> None:
    run_id = await _saved_run(app_cfg)
    response = await client.get(f"/api/runs/{run_id}/pdf")
    assert response.status_code == 200
    assert "iron-air-batteries-" in response.headers["content-disposition"]


async def test_unknown_runs_are_not_found(client: httpx.AsyncClient) -> None:
    for path in (f"/api/runs/{RUN_ID}", f"/api/runs/{RUN_ID}/report", "/api/runs/..%2f..%2fx/pdf"):
        assert (await client.get(path)).status_code in (404, 405)


async def test_the_page_is_served(client: httpx.AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    assert "Quarry" in response.text
