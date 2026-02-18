from __future__ import annotations

import time

from veldra.gui.job_store import GuiJobStore
from veldra.gui.types import GuiRunResult, RunInvocation
from veldra.gui.worker import GuiWorkerPool


def _wait_until(predicate, timeout_sec: float = 3.0) -> None:
    start = time.time()
    while time.time() - start < timeout_sec:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timeout while waiting for worker pool result.")


def test_worker_pool_processes_jobs_without_loss(monkeypatch, tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(
        "veldra.gui.worker.run_action",
        lambda _inv, **_kw: GuiRunResult(success=True, message="done", payload={}),
    )
    pool = GuiWorkerPool(store, worker_count=3, poll_interval_sec=0.05)
    pool.start()
    try:
        queued = [store.enqueue_job(RunInvocation(action="fit")) for _ in range(8)]
        def _all_done() -> bool:
            for job in queued:
                loaded = store.get_job(job.job_id)
                if loaded is None or loaded.status != "succeeded":
                    return False
            return True

        _wait_until(_all_done)
    finally:
        pool.stop()


def test_worker_pool_stop_prevents_new_claims(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    pool = GuiWorkerPool(store, worker_count=2, poll_interval_sec=0.05)
    pool.start()
    pool.stop()

    queued = store.enqueue_job(RunInvocation(action="fit"))
    time.sleep(0.2)
    loaded = store.get_job(queued.job_id)
    assert loaded is not None
    assert loaded.status == "queued"


def test_worker_pool_worker_count_one_is_supported(monkeypatch, tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(
        "veldra.gui.worker.run_action",
        lambda _inv, **_kw: GuiRunResult(success=False, message="bad", payload={}),
    )
    pool = GuiWorkerPool(store, worker_count=1, poll_interval_sec=0.05)
    pool.start()
    try:
        queued = store.enqueue_job(RunInvocation(action="fit"))
        _wait_until(lambda: (store.get_job(queued.job_id) or queued).status == "failed")
    finally:
        pool.stop()
