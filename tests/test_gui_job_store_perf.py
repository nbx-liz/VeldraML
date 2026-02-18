from __future__ import annotations

from time import perf_counter

from veldra.gui.job_store import GuiJobStore
from veldra.gui.types import RunInvocation


def test_list_jobs_page_p95_under_threshold(tmp_path) -> None:
    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    for idx in range(10_000):
        action = "fit" if idx % 2 == 0 else "tune"
        store.enqueue_job(RunInvocation(action=action))

    timings_ms: list[float] = []
    for _ in range(10):
        started = perf_counter()
        rows, total = store.list_jobs_page(limit=50, offset=0, status="queued", action="fit")
        elapsed_ms = (perf_counter() - started) * 1000.0
        timings_ms.append(elapsed_ms)
        assert total >= 5_000
        assert len(rows) == 50

    timings_ms.sort()
    p95 = timings_ms[int(len(timings_ms) * 0.95) - 1]
    assert p95 < 100.0
