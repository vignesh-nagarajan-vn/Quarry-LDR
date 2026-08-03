"""Run lifecycle round trips through SQLite; stages checkpoint and resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from quarry_ldr.state import RunStatus, RunStore, Stage, StageStatus, new_run_id


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.db"


async def test_run_round_trip(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        run = await store.create_run("test topic", {"chunk": {"target_tokens": 512}})
        assert run.status is RunStatus.CREATED
        assert run.config_snapshot["chunk"]["target_tokens"] == 512

        fetched = await store.get_run(run.run_id)
        assert fetched == run

        await store.set_run_status(run.run_id, RunStatus.RUNNING)
        await store.set_run_iteration(run.run_id, 2)
        updated = await store.get_run(run.run_id)
        assert updated.status is RunStatus.RUNNING
        assert updated.iteration == 2
        assert updated.updated_at >= run.updated_at


async def test_persistence_across_reopen(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        run = await store.create_run("persist me", {})
        await store.start_stage(run.run_id, Stage.PLAN)
        await store.complete_stage(run.run_id, Stage.PLAN, {"sub_questions": 9})

    async with RunStore(db_path) as store:
        fetched = await store.get_run(run.run_id)
        assert fetched.topic == "persist me"
        stage = await store.get_stage(run.run_id, Stage.PLAN)
        assert stage is not None
        assert stage.status is StageStatus.COMPLETED
        assert stage.payload == {"sub_questions": 9}
        assert stage.finished_at is not None


async def test_stage_lifecycle_and_resume_point(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        run = await store.create_run("resumable", {})
        await store.start_stage(run.run_id, Stage.PLAN)
        await store.complete_stage(run.run_id, Stage.PLAN, {"n": 1})
        await store.start_stage(run.run_id, Stage.SEARCH)
        await store.complete_stage(run.run_id, Stage.SEARCH, {"urls": 40})
        # Crash mid-fetch: FETCH started but never completed.
        await store.start_stage(run.run_id, Stage.FETCH)

        latest = await store.latest_completed_stage(run.run_id)
        assert latest is not None
        assert latest.stage is Stage.SEARCH
        assert latest.payload == {"urls": 40}

        all_stages = await store.stages(run.run_id)
        assert [s.stage for s in all_stages] == [Stage.PLAN, Stage.SEARCH, Stage.FETCH]
        assert all_stages[-1].status is StageStatus.RUNNING


async def test_fail_stage_and_retry_resets_row(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        run = await store.create_run("flaky", {})
        await store.start_stage(run.run_id, Stage.FETCH, iteration=1)
        failed = await store.fail_stage(run.run_id, Stage.FETCH, "timeout on example.org", 1)
        assert failed.status is StageStatus.FAILED
        assert failed.error == "timeout on example.org"

        retried = await store.start_stage(run.run_id, Stage.FETCH, iteration=1)
        assert retried.status is StageStatus.RUNNING
        assert retried.error is None
        assert retried.payload == {}
        assert retried.finished_at is None


async def test_iterations_are_separate_rows(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        run = await store.create_run("loops", {})
        for iteration in (0, 1, 2):
            await store.start_stage(run.run_id, Stage.SEARCH, iteration)
            await store.complete_stage(run.run_id, Stage.SEARCH, {"iter": iteration}, iteration)
        rows = await store.stages(run.run_id)
        assert len(rows) == 3
        assert [r.payload["iter"] for r in rows] == [0, 1, 2]


async def test_unknown_run_and_unstarted_stage_raise(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        with pytest.raises(KeyError):
            await store.get_run("nope")
        with pytest.raises(KeyError):
            await store.set_run_status("nope", RunStatus.FAILED)
        run = await store.create_run("x", {})
        with pytest.raises(KeyError):
            await store.complete_stage(run.run_id, Stage.RENDER, {})


async def test_list_runs_newest_first(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        first = await store.create_run("first", {})
        second = await store.create_run("second", {})
        runs = await store.list_runs()
        ids = [r.run_id for r in runs]
        assert set(ids) == {first.run_id, second.run_id}
        assert runs[0].created_at >= runs[-1].created_at


async def test_ledger_entries_round_trip(db_path: Path) -> None:
    async with RunStore(db_path) as store:
        run = await store.create_run("costly", {})
        await store.append_ledger_entry(run.run_id, {"model": "claude-opus-5", "cost_usd": 0.35})
        await store.append_ledger_entry(run.run_id, {"model": "claude-sonnet-5", "cost_usd": 0.02})
        entries = await store.ledger_entries(run.run_id)
        assert len(entries) == 2
        assert entries[0]["model"] == "claude-opus-5"


def test_new_run_id_shape() -> None:
    run_id = new_run_id()
    assert len(run_id) == 12
    assert run_id != new_run_id()
