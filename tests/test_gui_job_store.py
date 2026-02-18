from __future__ import annotations

import sqlite3

import pytest

from veldra.gui.job_store import GuiJobStore, _decode_invocation, _decode_result
from veldra.gui.types import GuiRunResult, RunInvocation


def test_job_store_enqueue_claim_and_complete(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    assert store.claim_next_job() is None
    queued = store.enqueue_job(RunInvocation(action="fit", config_yaml="config_version: 1\n"))
    assert queued.status == "queued"
    assert queued.priority == "normal"
    assert queued.progress_pct == 0.0

    claimed = store.claim_next_job()
    assert claimed is not None
    assert claimed.job_id == queued.job_id
    assert claimed.status == "running"

    done = store.mark_succeeded(
        queued.job_id,
        GuiRunResult(success=True, message="ok", payload={"k": 1}),
    )
    assert done is not None
    assert done.status == "succeeded"
    assert done.result is not None
    assert done.result.payload["k"] == 1
    assert done.progress_pct == 100.0
    assert done.current_step == "completed"
    assert len(store.list_jobs()) >= 1


def test_job_store_cancel_contract(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    queued = store.enqueue_job(RunInvocation(action="fit"))
    canceled = store.request_cancel(queued.job_id)
    assert canceled is not None
    assert canceled.status == "canceled"

    running = store.enqueue_job(RunInvocation(action="tune"))
    claim = store.claim_next_job()
    assert claim is not None
    requested = store.request_cancel(running.job_id)
    assert requested is not None
    assert requested.status == "cancel_requested"
    assert requested.cancel_requested is True


def test_job_store_persistence_across_instances(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store_a = GuiJobStore(path)
    first = store_a.enqueue_job(RunInvocation(action="estimate_dr"))
    store_b = GuiJobStore(path)
    loaded = store_b.get_job(first.job_id)
    assert loaded is not None
    assert loaded.job_id == first.job_id
    assert loaded.status == "queued"


def test_job_store_additional_branches(tmp_path, monkeypatch) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    queued = store.enqueue_job(RunInvocation(action="fit"))
    running = store.mark_running(queued.job_id)
    assert running is not None
    assert running.status == "running"

    failed = store.mark_failed(queued.job_id, message="x", payload={"r": 1})
    assert failed is not None
    assert failed.status == "failed"

    terminal_cancel = store.request_cancel(queued.job_id)
    assert terminal_cancel is not None
    assert terminal_cancel.status == "failed"

    canceled = store.mark_canceled(queued.job_id, message="c")
    assert canceled is not None
    assert canceled.status == "canceled"
    assert canceled.result is not None
    assert canceled.result.message == "c"

    assert store.get_job("missing") is None
    assert store.request_cancel("missing") is None

    queued2 = store.enqueue_job(RunInvocation(action="tune"))
    listed = store.list_jobs(status="queued", limit=10)
    assert any(item.job_id == queued2.job_id for item in listed)

    with pytest.raises(ValueError):
        _decode_invocation('["not","dict"]')
    with pytest.raises(ValueError):
        _decode_result('["not","dict"]')

    monkeypatch.setattr(store, "get_job", lambda _job_id: None)
    with pytest.raises(RuntimeError, match="Failed to read queued job"):
        store.enqueue_job(RunInvocation(action="fit"))


def test_job_store_get_jobs_and_delete_jobs_contract(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    first = store.enqueue_job(RunInvocation(action="fit"))
    second = store.enqueue_job(RunInvocation(action="tune"))

    assert store.get_jobs([]) == []

    selected = store.get_jobs([first.job_id, "", second.job_id])
    ids = {item.job_id for item in selected}
    assert first.job_id in ids
    assert second.job_id in ids

    assert store.delete_jobs([]) == 0
    deleted = store.delete_jobs([first.job_id, "", "missing-id"])
    assert deleted == 1
    assert store.get_job(first.job_id) is None
    assert store.get_job(second.job_id) is not None


def test_job_store_priority_claim_order_and_update(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    low = store.enqueue_job(RunInvocation(action="fit", priority="low"))
    high = store.enqueue_job(RunInvocation(action="fit", priority="high"))
    normal = store.enqueue_job(RunInvocation(action="fit", priority="normal"))

    claimed_1 = store.claim_next_job()
    claimed_2 = store.claim_next_job()
    claimed_3 = store.claim_next_job()

    assert claimed_1 is not None and claimed_1.job_id == high.job_id
    assert claimed_2 is not None and claimed_2.job_id == normal.job_id
    assert claimed_3 is not None and claimed_3.job_id == low.job_id

    queued = store.enqueue_job(RunInvocation(action="tune", priority="low"))
    updated = store.set_job_priority(queued.job_id, "high")
    assert updated is not None
    assert updated.priority == "high"


def test_job_store_set_priority_rejects_non_queued_status(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    queued = store.enqueue_job(RunInvocation(action="fit"))
    claimed = store.claim_next_job()
    assert claimed is not None
    with pytest.raises(ValueError, match="queued jobs"):
        store.set_job_priority(queued.job_id, "high")


def test_job_store_migrates_legacy_schema_with_missing_priority(tmp_path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                invocation_json TEXT NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                started_at_utc TEXT NULL,
                finished_at_utc TEXT NULL,
                result_json TEXT NULL,
                error_message TEXT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    store = GuiJobStore(db_path)
    queued = store.enqueue_job(RunInvocation(action="fit"))
    assert queued.priority == "normal"
    assert queued.progress_pct == 0.0


def test_job_store_update_progress_and_logs(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    queued = store.enqueue_job(RunInvocation(action="fit"))

    updated = store.update_progress(queued.job_id, 42.5, step="train_fold_1")
    assert updated is not None
    assert updated.progress_pct == 42.5
    assert updated.current_step == "train_fold_1"

    store.append_job_log(
        queued.job_id,
        level="info",
        message="started",
        payload={"x": 1},
    )
    store.append_job_log(
        queued.job_id,
        level="error",
        message="failed",
        payload={"reason": "boom"},
    )
    logs = store.list_job_logs(queued.job_id, limit=10)
    assert len(logs) == 2
    assert logs[0].message == "started"
    assert logs[1].level == "ERROR"


def test_job_store_log_retention_keeps_recent_entries(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    queued = store.enqueue_job(RunInvocation(action="fit"))

    for idx in range(store.LOG_RETENTION_PER_JOB + 5):
        store.append_job_log(
            queued.job_id,
            level="INFO",
            message=f"line_{idx}",
            payload={"i": idx},
        )

    logs = store.list_job_logs(queued.job_id, limit=store.LOG_RETENTION_PER_JOB)
    assert len(logs) == store.LOG_RETENTION_PER_JOB
    assert any(row.message == "log_retention_applied" for row in logs)
