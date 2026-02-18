"""Background worker for GUI async jobs."""

from __future__ import annotations

import logging
import threading
import time

from veldra.api.logging import log_event
from veldra.gui.job_store import GuiJobStore
from veldra.gui.services import run_action

LOGGER = logging.getLogger("veldra.gui.worker")


class GuiWorker:
    """Single-worker loop that executes queued GUI jobs."""

    def __init__(
        self,
        store: GuiJobStore,
        *,
        poll_interval_sec: float = 0.5,
        worker_name: str | None = None,
    ) -> None:
        self._store = store
        self._poll_interval_sec = max(0.05, float(poll_interval_sec))
        self._worker_name = str(worker_name or "veldra-gui-worker")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=self._worker_name,
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_sec: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
        self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            claimed = self._store.claim_next_job()
            if claimed is None:
                self._stop_event.wait(self._poll_interval_sec)
                continue

            log_event(
                LOGGER,
                logging.INFO,
                "gui job updated",
                run_id=claimed.job_id,
                artifact_path=claimed.invocation.artifact_path,
                task_type=claimed.action,
                event="gui job updated",
                status="running",
                action=claimed.action,
            )
            self._store.update_progress(claimed.job_id, 1.0, step="started")
            self._store.append_job_log(
                claimed.job_id,
                level="INFO",
                message="worker_started",
                payload={"worker": self._worker_name, "action": claimed.action},
            )

            try:
                result = run_action(
                    claimed.invocation,
                    job_id=claimed.job_id,
                    job_store=self._store,
                )
            except Exception as exc:  # pragma: no cover - defensive branch.
                updated = self._store.mark_failed(
                    claimed.job_id,
                    message=f"Unhandled worker error: {exc}",
                    payload={},
                )
                if updated is not None:
                    log_event(
                        LOGGER,
                        logging.ERROR,
                        "gui job updated",
                        run_id=updated.job_id,
                        artifact_path=updated.invocation.artifact_path,
                        task_type=updated.action,
                        event="gui job updated",
                        status=updated.status,
                        action=updated.action,
                    )
                continue

            if result.success:
                updated = self._store.mark_succeeded(claimed.job_id, result)
                self._store.append_job_log(
                    claimed.job_id,
                    level="INFO",
                    message="worker_completed",
                    payload={"status": "succeeded"},
                )
            else:
                updated = self._store.mark_failed(
                    claimed.job_id,
                    message=result.message,
                    payload=result.payload,
                )
                self._store.append_job_log(
                    claimed.job_id,
                    level="ERROR",
                    message="worker_completed",
                    payload={"status": "failed", "error": result.message},
                )

            if updated is not None:
                log_event(
                    LOGGER,
                    logging.INFO if updated.status == "succeeded" else logging.WARNING,
                    "gui job updated",
                    run_id=updated.job_id,
                    artifact_path=updated.invocation.artifact_path,
                    task_type=updated.action,
                    event="gui job updated",
                    status=updated.status,
                    action=updated.action,
                    cancel_requested=updated.cancel_requested,
                )
            time.sleep(0.001)


class GuiWorkerPool:
    """Manage a pool of GUI workers."""

    def __init__(
        self,
        store: GuiJobStore,
        *,
        worker_count: int = 1,
        poll_interval_sec: float = 0.5,
    ) -> None:
        safe_count = max(1, int(worker_count))
        self._workers = [
            GuiWorker(
                store,
                poll_interval_sec=poll_interval_sec,
                worker_name=f"veldra-gui-worker-{idx + 1}",
            )
            for idx in range(safe_count)
        ]

    @property
    def is_running(self) -> bool:
        return any(worker.is_running for worker in self._workers)

    def start(self) -> None:
        for worker in self._workers:
            worker.start()

    def stop(self, *, timeout_sec: float = 2.0) -> None:
        for worker in self._workers:
            worker.stop(timeout_sec=timeout_sec)
