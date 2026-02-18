from __future__ import annotations

from veldra.gui.job_store import GuiJobStore
from veldra.gui.services import (
    cancel_run_job,
    get_run_job,
    list_run_jobs,
    retry_run_job,
    set_gui_runtime,
    stop_gui_runtime,
    submit_run_job,
)
from veldra.gui.types import RunInvocation


class _StubWorker:
    def __init__(self) -> None:
        self.started = 0

    def start(self) -> None:
        self.started += 1


def test_submit_list_get_cancel_job(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    worker = _StubWorker()
    set_gui_runtime(job_store=store, worker=worker)

    submitted = submit_run_job(RunInvocation(action="fit"))
    assert submitted.status == "queued"
    assert worker.started == 1

    listed = list_run_jobs(limit=10)
    assert len(listed) == 1
    assert listed[0].job_id == submitted.job_id

    loaded = get_run_job(submitted.job_id)
    assert loaded is not None
    assert loaded.status == "queued"

    canceled = cancel_run_job(submitted.job_id)
    assert canceled.status == "canceled"
    updated = get_run_job(submitted.job_id)
    assert updated is not None
    assert updated.status == "canceled"
    retried = retry_run_job(submitted.job_id)
    assert retried.status == "queued"
    retried_job = get_run_job(retried.job_id)
    assert retried_job is not None
    assert retried_job.retry_parent_job_id == submitted.job_id
    stop_gui_runtime()
