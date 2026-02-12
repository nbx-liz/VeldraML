from __future__ import annotations

import pytest

from veldra.gui.job_store import GuiJobStore, _decode_invocation, _decode_result
from veldra.gui.types import GuiRunResult, RunInvocation


def test_job_store_enqueue_claim_and_complete(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    assert store.claim_next_job() is None
    queued = store.enqueue_job(RunInvocation(action="fit", config_yaml="config_version: 1\n"))
    assert queued.status == "queued"

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
