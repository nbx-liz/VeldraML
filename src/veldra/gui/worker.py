"""Background worker for GUI async jobs."""

from __future__ import annotations

import logging
import math
import threading
import time

from veldra.api.logging import log_event
from veldra.gui.job_store import GuiJobStore
from veldra.gui.services import CanceledByUser, run_action
from veldra.gui.types import RetryPolicy

LOGGER = logging.getLogger("veldra.gui.worker")


def _retry_delay_sec(retry_count: int, policy: RetryPolicy | None) -> float:
    if policy is None:
        return 0.0
    base = max(0.0, float(policy.base_delay_sec))
    cap = max(base, float(policy.max_delay_sec))
    return min(cap, base * math.pow(2.0, max(0, int(retry_count) - 1)))


def _normalize_policy(policy: RetryPolicy | dict[str, object] | None) -> RetryPolicy | None:
    if policy is None:
        return None
    if isinstance(policy, RetryPolicy):
        return policy
    if isinstance(policy, dict):
        try:
            return RetryPolicy(**policy)
        except Exception:
            return None
    return None


def _should_auto_retry(
    *,
    retry_count: int,
    error_kind: str,
    policy: RetryPolicy | None,
) -> bool:
    if policy is None:
        return False
    if int(policy.max_retries) <= int(retry_count):
        return False
    retry_on = {str(k).strip().lower() for k in policy.retry_on}
    return str(error_kind).strip().lower() in retry_on


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
            except CanceledByUser:
                updated = self._store.mark_canceled_from_request(
                    claimed.job_id,
                    message="Canceled by user request.",
                )
                self._store.append_job_log(
                    claimed.job_id,
                    level="WARNING",
                    message="worker_completed",
                    payload={"status": "canceled"},
                )
                if updated is not None:
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "gui job updated",
                        run_id=updated.job_id,
                        artifact_path=updated.invocation.artifact_path,
                        task_type=updated.action,
                        event="gui job updated",
                        status=updated.status,
                        action=updated.action,
                        cancel_requested=updated.cancel_requested,
                    )
                continue
            except Exception as exc:  # pragma: no cover - defensive branch.
                updated = self._store.mark_failed(
                    claimed.job_id,
                    message=f"Unhandled worker error: {exc}",
                    payload={},
                    error_kind="unknown",
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
                if self._store.is_cancel_requested(claimed.job_id):
                    updated = self._store.mark_canceled_from_request(
                        claimed.job_id,
                        message="Canceled by user request.",
                    )
                    self._store.append_job_log(
                        claimed.job_id,
                        level="WARNING",
                        message="worker_completed",
                        payload={"status": "canceled"},
                    )
                else:
                    updated = self._store.mark_succeeded(claimed.job_id, result)
                    self._store.append_job_log(
                        claimed.job_id,
                        level="INFO",
                        message="worker_completed",
                        payload={"status": "succeeded"},
                    )
            else:
                error_kind = str(result.payload.get("error_kind", "unknown")).strip().lower()
                updated = self._store.mark_failed(
                    claimed.job_id,
                    message=result.message,
                    payload=result.payload,
                    error_kind=error_kind,
                )
                self._store.append_job_log(
                    claimed.job_id,
                    level="ERROR",
                    message="worker_completed",
                    payload={"status": "failed", "error": result.message, "error_kind": error_kind},
                )
                policy = _normalize_policy(claimed.invocation.retry_policy)
                if (
                    updated is not None
                    and _should_auto_retry(
                        retry_count=updated.retry_count,
                        error_kind=error_kind,
                        policy=policy,
                    )
                ):
                    delay = _retry_delay_sec(updated.retry_count + 1, policy)
                    if delay > 0:
                        self._store.append_job_log(
                            claimed.job_id,
                            level="INFO",
                            message="retry_backoff_sleep",
                            payload={"delay_sec": delay},
                        )
                        self._stop_event.wait(delay)
                    retry_job = self._store.create_retry_job(
                        claimed.job_id,
                        reason=f"auto_retry:{error_kind}",
                        policy=policy,
                    )
                    self._store.append_job_log(
                        claimed.job_id,
                        level="INFO",
                        message="retry_auto_enqueued",
                        payload={"retry_job_id": retry_job.job_id},
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
