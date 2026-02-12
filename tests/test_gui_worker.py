from __future__ import annotations

import time

from veldra.gui.job_store import GuiJobStore
from veldra.gui.types import GuiRunResult, RunInvocation
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
        lambda _inv: GuiRunResult(success=True, message="done", payload={"x": 1}),
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
        lambda _inv: GuiRunResult(success=False, message="bad", payload={"reason": "x"}),
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
