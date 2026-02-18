from __future__ import annotations

import time

from veldra.gui.job_store import GuiJobStore
from veldra.gui.services import CanceledByUser
from veldra.gui.types import GuiRunResult, RetryPolicy, RunInvocation
from veldra.gui.worker import GuiWorker


def _wait_until(predicate, timeout_sec: float = 2.0) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timeout while waiting for worker result.")


def test_worker_processes_job_success(monkeypatch, tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(
        "veldra.gui.worker.run_action",
        lambda _inv, **_kw: GuiRunResult(success=True, message="done", payload={"x": 1}),
    )
    worker = GuiWorker(store, poll_interval_sec=0.05)
    worker.start()
    try:
        queued = store.enqueue_job(RunInvocation(action="fit"))
        _wait_until(lambda: (store.get_job(queued.job_id) or queued).status == "succeeded")
        done = store.get_job(queued.job_id)
        assert done is not None
        assert done.status == "succeeded"
        assert done.result is not None
        assert done.result.payload["x"] == 1
    finally:
        worker.stop()


def test_worker_processes_job_failure(monkeypatch, tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(
        "veldra.gui.worker.run_action",
        lambda _inv, **_kw: GuiRunResult(success=False, message="bad", payload={"reason": "x"}),
    )
    worker = GuiWorker(store, poll_interval_sec=0.05)
    worker.start()
    try:
        queued = store.enqueue_job(RunInvocation(action="fit"))
        _wait_until(lambda: (store.get_job(queued.job_id) or queued).status == "failed")
        done = store.get_job(queued.job_id)
        assert done is not None
        assert done.status == "failed"
        assert done.error_message == "bad"
    finally:
        worker.stop()


def test_worker_marks_canceled_on_canceled_by_user(monkeypatch, tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")

    def _raise_cancel(_inv, **_kw):
        raise CanceledByUser("Canceled by user request.")

    monkeypatch.setattr("veldra.gui.worker.run_action", _raise_cancel)
    worker = GuiWorker(store, poll_interval_sec=0.05)
    worker.start()
    try:
        queued = store.enqueue_job(RunInvocation(action="fit"))
        _wait_until(lambda: (store.get_job(queued.job_id) or queued).status == "canceled")
        done = store.get_job(queued.job_id)
        assert done is not None
        assert done.status == "canceled"
    finally:
        worker.stop()


def test_worker_does_not_auto_retry_when_policy_zero(monkeypatch, tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(
        "veldra.gui.worker.run_action",
        lambda _inv, **_kw: GuiRunResult(
            success=False,
            message="timed out",
            payload={"error_kind": "timeout", "next_steps": []},
        ),
    )
    worker = GuiWorker(store, poll_interval_sec=0.05)
    worker.start()
    try:
        queued = store.enqueue_job(
            RunInvocation(
                action="fit",
                retry_policy=RetryPolicy(max_retries=0, retry_on=("timeout",)),
            )
        )
        _wait_until(lambda: (store.get_job(queued.job_id) or queued).status == "failed")
        jobs = store.list_jobs(limit=10)
        assert len(jobs) == 1
        assert jobs[0].job_id == queued.job_id
    finally:
        worker.stop()


def test_worker_prioritizes_cancel_request_over_success(monkeypatch, tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")

    def _success_with_cancel(_inv, *, job_id, **_kw):
        store.request_cancel(job_id)
        return GuiRunResult(success=True, message="ok", payload={})

    monkeypatch.setattr("veldra.gui.worker.run_action", _success_with_cancel)
    worker = GuiWorker(store, poll_interval_sec=0.05)
    worker.start()
    try:
        queued = store.enqueue_job(RunInvocation(action="fit"))
        _wait_until(lambda: (store.get_job(queued.job_id) or queued).status == "canceled")
        done = store.get_job(queued.job_id)
        assert done is not None
        assert done.status == "canceled"
    finally:
        worker.stop()
