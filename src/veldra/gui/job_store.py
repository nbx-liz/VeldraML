"""SQLite-backed job persistence for GUI async runs."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from veldra.gui.types import (
    GuiJobLogRecord,
    GuiJobPriority,
    GuiJobRecord,
    GuiRunResult,
    RetryPolicy,
    RunInvocation,
)

TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
_PRIORITY_TO_INT: dict[GuiJobPriority, int] = {"low": 10, "normal": 50, "high": 90}
_ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
LOGGER = logging.getLogger("veldra.gui.job_store")


def _priority_to_int(priority: str | None) -> int:
    key = str(priority or "normal").strip().lower()
    if key == "low":
        return _PRIORITY_TO_INT["low"]
    if key == "high":
        return _PRIORITY_TO_INT["high"]
    return _PRIORITY_TO_INT["normal"]


def _priority_from_int(value: Any) -> GuiJobPriority:
    try:
        score = int(value)
    except Exception:
        return "normal"
    if score >= 80:
        return "high"
    if score <= 20:
        return "low"
    return "normal"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _decode_invocation(raw: str) -> RunInvocation:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Invalid invocation payload in job store.")
    retry_policy = payload.get("retry_policy")
    if isinstance(retry_policy, dict):
        payload["retry_policy"] = RetryPolicy(**retry_policy)
    return RunInvocation(**payload)


def _coerce_retry_policy(policy: RetryPolicy | dict[str, Any] | None) -> RetryPolicy | None:
    if policy is None:
        return None
    if isinstance(policy, RetryPolicy):
        return policy
    if isinstance(policy, dict):
        return RetryPolicy(**policy)
    return None


def _decode_result(raw: str | None) -> GuiRunResult | None:
    if raw is None:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Invalid result payload in job store.")
    return GuiRunResult(**payload)


def _row_to_record(row: sqlite3.Row) -> GuiJobRecord:
    return GuiJobRecord(
        job_id=str(row["job_id"]),
        status=str(row["status"]),
        action=str(row["action"]),
        created_at_utc=str(row["created_at_utc"]),
        updated_at_utc=str(row["updated_at_utc"]),
        invocation=_decode_invocation(str(row["invocation_json"])),
        priority=_priority_from_int(row["priority"]),
        progress_pct=float(row["progress_pct"]),
        current_step=(str(row["current_step"]) if row["current_step"] is not None else None),
        cancel_requested=bool(int(row["cancel_requested"])),
        started_at_utc=(str(row["started_at_utc"]) if row["started_at_utc"] is not None else None),
        finished_at_utc=(
            str(row["finished_at_utc"]) if row["finished_at_utc"] is not None else None
        ),
        result=_decode_result(row["result_json"]),
        error_message=(str(row["error_message"]) if row["error_message"] is not None else None),
        retry_count=int(row["retry_count"] or 0),
        retry_parent_job_id=(
            str(row["retry_parent_job_id"]) if row["retry_parent_job_id"] is not None else None
        ),
        last_error_kind=(
            str(row["last_error_kind"]) if row["last_error_kind"] is not None else None
        ),
    )


class GuiJobStore:
    """Persist GUI run jobs in SQLite."""

    LOG_RETENTION_PER_JOB = 10_000
    SLOW_QUERY_MS = 100.0

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    action TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 50,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    invocation_json TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    progress_pct REAL NOT NULL DEFAULT 0.0,
                    current_step TEXT NULL,
                    started_at_utc TEXT NULL,
                    finished_at_utc TEXT NULL,
                    result_json TEXT NULL,
                    error_message TEXT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    retry_parent_job_id TEXT NULL,
                    last_error_kind TEXT NULL
                )
                """
            )
            columns = conn.execute("PRAGMA table_info(jobs)").fetchall()
            column_names = {str(row["name"]) for row in columns}
            if "priority" not in column_names:
                conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 50")
            if "progress_pct" not in column_names:
                conn.execute("ALTER TABLE jobs ADD COLUMN progress_pct REAL NOT NULL DEFAULT 0.0")
            if "current_step" not in column_names:
                conn.execute("ALTER TABLE jobs ADD COLUMN current_step TEXT NULL")
            if "retry_count" not in column_names:
                conn.execute("ALTER TABLE jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
            if "retry_parent_job_id" not in column_names:
                conn.execute("ALTER TABLE jobs ADD COLUMN retry_parent_job_id TEXT NULL")
            if "last_error_kind" not in column_names:
                conn.execute("ALTER TABLE jobs ADD COLUMN last_error_kind TEXT NULL")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs_archive (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    action TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 50,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    invocation_json TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    progress_pct REAL NOT NULL DEFAULT 0.0,
                    current_step TEXT NULL,
                    started_at_utc TEXT NULL,
                    finished_at_utc TEXT NULL,
                    result_json TEXT NULL,
                    error_message TEXT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    retry_parent_job_id TEXT NULL,
                    last_error_kind TEXT NULL,
                    archived_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at_utc)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status_priority_created
                ON jobs(status, priority DESC, created_at_utc ASC)
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_job_logs_job_seq ON job_logs(job_id, seq)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_action_created ON jobs(action, created_at_utc)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at_utc)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_archive_finished
                ON jobs_archive(finished_at_utc, archived_at_utc)
                """
            )

    def _track_query(
        self,
        *,
        query_name: str,
        started_at: float,
        row_count: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> float:
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        payload = {
            "query_name": query_name,
            "elapsed_ms": round(elapsed_ms, 3),
            "db_path": str(self.db_path),
        }
        if row_count is not None:
            payload["row_count"] = int(row_count)
        if extra:
            payload.update(extra)
        if elapsed_ms >= self.SLOW_QUERY_MS:
            LOGGER.warning("gui_db_slow_query", extra=payload)
        else:
            LOGGER.debug("gui_db_query", extra=payload)
        return elapsed_ms

    def enqueue_job(self, invocation: RunInvocation) -> GuiJobRecord:
        now = _utcnow_iso()
        job_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, status, action, priority, created_at_utc, updated_at_utc,
                    invocation_json, cancel_requested, progress_pct, current_step,
                    started_at_utc, finished_at_utc, result_json, error_message,
                    retry_count, retry_parent_job_id, last_error_kind
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, 0, 0.0, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL)
                """,
                (
                    job_id,
                    "queued",
                    invocation.action.strip().lower(),
                    _priority_to_int(invocation.priority),
                    now,
                    now,
                    json.dumps(asdict(invocation), ensure_ascii=False),
                ),
            )
        record = self.get_job(job_id)
        if record is None:
            raise RuntimeError("Failed to read queued job.")
        return record

    def request_cancel(self, job_id: str) -> GuiJobRecord | None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None

            status = str(row["status"])
            if status in TERMINAL_STATUSES:
                conn.execute("COMMIT")
                return _row_to_record(row)

            if status == "queued":
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'canceled',
                        cancel_requested = 1,
                        progress_pct = 100.0,
                        current_step = 'canceled',
                        updated_at_utc = ?,
                        finished_at_utc = ?
                    WHERE job_id = ?
                    """,
                    (now, now, job_id),
                )
            elif status == "running":
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'cancel_requested',
                        cancel_requested = 1,
                        current_step = 'cancellation_requested',
                        updated_at_utc = ?
                    WHERE job_id = ?
                    """,
                    (now, job_id),
                )
            conn.execute("COMMIT")
        return self.get_job(job_id)

    def claim_next_job(self) -> GuiJobRecord | None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at_utc ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            job_id = str(row["job_id"])
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    current_step = COALESCE(current_step, 'running'),
                    started_at_utc = COALESCE(started_at_utc, ?),
                    updated_at_utc = ?
                WHERE job_id = ?
                """,
                (now, now, job_id),
            )
            conn.execute("COMMIT")
        return self.get_job(job_id)

    def set_job_priority(self, job_id: str, priority: str | GuiJobPriority) -> GuiJobRecord | None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            status = str(row["status"])
            if status != "queued":
                conn.execute("COMMIT")
                raise ValueError(f"Priority can only be changed for queued jobs (status={status}).")
            conn.execute(
                """
                UPDATE jobs
                SET priority = ?,
                    updated_at_utc = ?
                WHERE job_id = ?
                """,
                (_priority_to_int(priority), now, job_id),
            )
            conn.execute("COMMIT")
        return self.get_job(job_id)

    def mark_running(self, job_id: str) -> GuiJobRecord | None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    current_step = COALESCE(current_step, 'running'),
                    started_at_utc = COALESCE(started_at_utc, ?),
                    updated_at_utc = ?
                WHERE job_id = ?
                """,
                (now, now, job_id),
            )
        return self.get_job(job_id)

    def mark_succeeded(self, job_id: str, result: GuiRunResult) -> GuiJobRecord | None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    progress_pct = 100.0,
                    current_step = 'completed',
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    result_json = ?,
                    error_message = NULL
                WHERE job_id = ?
                """,
                (now, now, json.dumps(asdict(result), ensure_ascii=False), job_id),
            )
        return self.get_job(job_id)

    def mark_failed(
        self,
        job_id: str,
        *,
        message: str,
        payload: dict[str, Any] | None = None,
        error_kind: str | None = None,
    ) -> GuiJobRecord | None:
        now = _utcnow_iso()
        result = GuiRunResult(success=False, message=message, payload=payload or {})
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    progress_pct = 100.0,
                    current_step = 'failed',
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    result_json = ?,
                    error_message = ?,
                    last_error_kind = ?
                WHERE job_id = ?
                """,
                (
                    now,
                    now,
                    json.dumps(asdict(result), ensure_ascii=False),
                    message,
                    (str(error_kind).strip() if error_kind else None),
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def mark_canceled(self, job_id: str, *, message: str = "Canceled.") -> GuiJobRecord | None:
        now = _utcnow_iso()
        result = GuiRunResult(success=False, message=message, payload={})
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    cancel_requested = 1,
                    progress_pct = 100.0,
                    current_step = 'canceled',
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    result_json = ?,
                    error_message = NULL,
                    last_error_kind = 'cancel'
                WHERE job_id = ?
                """,
                (now, now, json.dumps(asdict(result), ensure_ascii=False), job_id),
            )
        return self.get_job(job_id)

    def mark_canceled_from_request(
        self,
        job_id: str,
        *,
        message: str = "Canceled by user.",
    ) -> GuiJobRecord | None:
        return self.mark_canceled(job_id, message=message)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested, status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return False
            return bool(int(row["cancel_requested"])) or str(row["status"]) == "cancel_requested"

    def create_retry_job(
        self,
        source_job_id: str,
        *,
        reason: str,
        policy: RetryPolicy | None,
    ) -> GuiJobRecord:
        now = _utcnow_iso()
        new_job_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            source_row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (source_job_id,),
            ).fetchone()
            if source_row is None:
                conn.execute("COMMIT")
                raise ValueError(f"Source job not found: {source_job_id}")
            source = _row_to_record(source_row)
            retry_invocation = RunInvocation(
                action=source.invocation.action,
                config_yaml=source.invocation.config_yaml,
                config_path=source.invocation.config_path,
                data_path=source.invocation.data_path,
                artifact_path=source.invocation.artifact_path,
                scenarios_path=source.invocation.scenarios_path,
                export_format=source.invocation.export_format,
                priority=source.invocation.priority,
                retry_policy=(
                    _coerce_retry_policy(policy)
                    or _coerce_retry_policy(source.invocation.retry_policy)
                ),
            )
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, status, action, priority, created_at_utc, updated_at_utc,
                    invocation_json, cancel_requested, progress_pct, current_step,
                    started_at_utc, finished_at_utc, result_json, error_message,
                    retry_count, retry_parent_job_id, last_error_kind
                )
                VALUES(
                    ?, 'queued', ?, ?, ?, ?, ?, 0, 0.0, 'retry_queued',
                    NULL, NULL, NULL, NULL, ?, ?, NULL
                )
                """,
                (
                    new_job_id,
                    retry_invocation.action.strip().lower(),
                    _priority_to_int(retry_invocation.priority),
                    now,
                    now,
                    json.dumps(asdict(retry_invocation), ensure_ascii=False),
                    int(source.retry_count) + 1,
                    source.job_id,
                ),
            )
            reason_payload = json.dumps(
                {"reason": str(reason or "manual_retry"), "next_job_id": new_job_id},
                ensure_ascii=False,
            )
            conn.execute(
                """
                INSERT INTO job_logs(job_id, seq, created_at_utc, level, message, payload_json)
                VALUES(
                    ?,
                    COALESCE((SELECT MAX(seq) + 1 FROM job_logs WHERE job_id = ?), 1),
                    ?, 'INFO', 'retry_enqueued', ?
                )
                """,
                (source.job_id, source.job_id, now, reason_payload),
            )
            conn.execute("COMMIT")
        retry_job = self.get_job(new_job_id)
        if retry_job is None:
            raise RuntimeError("Failed to read retry job.")
        return retry_job

    def update_progress(
        self,
        job_id: str,
        pct: float,
        step: str | None = None,
    ) -> GuiJobRecord | None:
        now = _utcnow_iso()
        safe_pct = max(0.0, min(float(pct), 100.0))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE jobs
                SET progress_pct = ?,
                    current_step = COALESCE(?, current_step),
                    updated_at_utc = ?
                WHERE job_id = ?
                """,
                (safe_pct, (step or None), now, job_id),
            )
            conn.execute("COMMIT")
        return self.get_job(job_id)

    def append_job_log(
        self,
        job_id: str,
        *,
        level: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        now = _utcnow_iso()
        safe_level = str(level or "INFO").strip().upper()
        if safe_level not in _ALLOWED_LOG_LEVELS:
            safe_level = "INFO"
        safe_message = str(message or "").strip()
        if not safe_message:
            safe_message = "(empty log)"
        payload_json = json.dumps(payload or {}, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT COUNT(*) AS cnt FROM job_logs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            current_count = int(current["cnt"]) if current is not None else 0
            dropped = 0
            if current_count + 1 > self.LOG_RETENTION_PER_JOB:
                needed_entries = 2
                dropped = (current_count + needed_entries) - self.LOG_RETENTION_PER_JOB
                conn.execute(
                    """
                    DELETE FROM job_logs
                    WHERE id IN (
                        SELECT id
                        FROM job_logs
                        WHERE job_id = ?
                        ORDER BY seq ASC
                        LIMIT ?
                    )
                    """,
                    (job_id, dropped),
                )

            seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM job_logs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            seq = int(seq_row["max_seq"]) + 1 if seq_row is not None else 1
            if dropped > 0:
                meta_payload = json.dumps({"dropped_count": dropped}, ensure_ascii=False)
                conn.execute(
                    """
                    INSERT INTO job_logs(job_id, seq, created_at_utc, level, message, payload_json)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, seq, now, "WARNING", "log_retention_applied", meta_payload),
                )
                seq += 1

            conn.execute(
                """
                INSERT INTO job_logs(job_id, seq, created_at_utc, level, message, payload_json)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (job_id, seq, now, safe_level, safe_message, payload_json),
            )
            conn.execute("COMMIT")

    def list_job_logs(
        self,
        job_id: str,
        *,
        limit: int = 200,
        after_seq: int | None = None,
    ) -> list[GuiJobLogRecord]:
        return self.list_job_logs_page(job_id, limit=limit, after_seq=after_seq)

    def list_job_logs_page(
        self,
        job_id: str,
        *,
        limit: int = 200,
        after_seq: int | None = None,
    ) -> list[GuiJobLogRecord]:
        safe_limit = max(1, min(int(limit), self.LOG_RETENTION_PER_JOB))
        started_at = perf_counter()
        with self._connect() as conn:
            if after_seq is None:
                rows = conn.execute(
                    """
                    SELECT job_id, seq, created_at_utc, level, message, payload_json
                    FROM job_logs
                    WHERE job_id = ?
                    ORDER BY seq DESC
                    LIMIT ?
                    """,
                    (job_id, safe_limit),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = conn.execute(
                    """
                    SELECT job_id, seq, created_at_utc, level, message, payload_json
                    FROM job_logs
                    WHERE job_id = ? AND seq > ?
                    ORDER BY seq ASC
                    LIMIT ?
                    """,
                    (job_id, int(after_seq), safe_limit),
                ).fetchall()
        self._track_query(
            query_name="list_job_logs_page",
            started_at=started_at,
            row_count=len(rows),
            extra={"job_id": job_id, "after_seq": after_seq, "limit": safe_limit},
        )
        return [self._row_to_log_record(row) for row in rows]

    def _row_to_log_record(self, row: sqlite3.Row) -> GuiJobLogRecord:
        raw_payload = row["payload_json"]
        payload: dict[str, Any] = {}
        if raw_payload is not None:
            try:
                decoded = json.loads(str(raw_payload))
                if isinstance(decoded, dict):
                    payload = decoded
            except Exception:
                payload = {"raw": str(raw_payload)}
        return GuiJobLogRecord(
            job_id=str(row["job_id"]),
            seq=int(row["seq"]),
            created_at_utc=str(row["created_at_utc"]),
            level=str(row["level"]),
            message=str(row["message"]),
            payload=payload,
        )

    def get_job(self, job_id: str) -> GuiJobRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_jobs(
        self,
        *,
        limit: int = 100,
        status: str | None = None,
    ) -> list[GuiJobRecord]:
        rows, _total = self.list_jobs_page(limit=limit, offset=0, status=status)
        return rows

    def list_jobs_page(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        action: str | None = None,
        query: str | None = None,
    ) -> tuple[list[GuiJobRecord], int]:
        safe_limit = max(1, min(int(limit), 1000))
        safe_offset = max(0, int(offset))
        where: list[str] = []
        params: list[Any] = []
        if status and str(status).strip():
            where.append("status = ?")
            params.append(str(status).strip())
        if action and str(action).strip():
            where.append("action = ?")
            params.append(str(action).strip().lower())
        if query and str(query).strip():
            q = f"%{str(query).strip().lower()}%"
            where.append(
                "("
                "LOWER(job_id) LIKE ? OR "
                "LOWER(invocation_json) LIKE ? OR "
                "LOWER(COALESCE(error_message, '')) LIKE ?"
                ")"
            )
            params.extend([q, q, q])
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        base_sql = f"FROM jobs{where_sql}"
        started_at = perf_counter()
        with self._connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS cnt {base_sql}",
                tuple(params),
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT * {base_sql}
                ORDER BY created_at_utc DESC
                LIMIT ? OFFSET ?
                """,
                tuple([*params, safe_limit, safe_offset]),
            ).fetchall()
        total_count = int(total_row["cnt"]) if total_row is not None else 0
        self._track_query(
            query_name="list_jobs_page",
            started_at=started_at,
            row_count=len(rows),
            extra={
                "total_count": total_count,
                "limit": safe_limit,
                "offset": safe_offset,
                "status": status,
                "action": action,
                "query": bool(query and str(query).strip()),
            },
        )
        return [_row_to_record(row) for row in rows], total_count

    def archive_jobs(self, *, cutoff_utc: str, batch_size: int = 200) -> int:
        safe_batch = max(1, min(int(batch_size), 5000))
        started_at = perf_counter()
        moved_count = 0
        archived_at = _utcnow_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            candidates = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('succeeded', 'failed', 'canceled')
                  AND finished_at_utc IS NOT NULL
                  AND finished_at_utc < ?
                ORDER BY finished_at_utc ASC
                LIMIT ?
                """,
                (cutoff_utc, safe_batch),
            ).fetchall()
            for row in candidates:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO jobs_archive(
                        job_id, status, action, priority, created_at_utc, updated_at_utc,
                        invocation_json, cancel_requested, progress_pct, current_step,
                        started_at_utc, finished_at_utc, result_json, error_message,
                        retry_count, retry_parent_job_id, last_error_kind, archived_at_utc
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row["job_id"]),
                        str(row["status"]),
                        str(row["action"]),
                        int(row["priority"]),
                        str(row["created_at_utc"]),
                        str(row["updated_at_utc"]),
                        str(row["invocation_json"]),
                        int(row["cancel_requested"]),
                        float(row["progress_pct"]),
                        row["current_step"],
                        row["started_at_utc"],
                        row["finished_at_utc"],
                        row["result_json"],
                        row["error_message"],
                        int(row["retry_count"] or 0),
                        row["retry_parent_job_id"],
                        row["last_error_kind"],
                        archived_at,
                    ),
                )
            ids = [str(row["job_id"]) for row in candidates]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM job_logs WHERE job_id IN ({placeholders})",
                    tuple(ids),
                )
                cur = conn.execute(
                    f"DELETE FROM jobs WHERE job_id IN ({placeholders})",
                    tuple(ids),
                )
                moved_count = int(cur.rowcount or 0)
            conn.execute("COMMIT")
        self._track_query(
            query_name="archive_jobs",
            started_at=started_at,
            row_count=moved_count,
            extra={"cutoff_utc": cutoff_utc, "batch_size": safe_batch},
        )
        return moved_count

    def purge_archived_jobs(self, *, cutoff_utc: str, batch_size: int = 200) -> int:
        safe_batch = max(1, min(int(batch_size), 5000))
        started_at = perf_counter()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            ids = conn.execute(
                """
                SELECT job_id
                FROM jobs_archive
                WHERE archived_at_utc < ?
                ORDER BY archived_at_utc ASC
                LIMIT ?
                """,
                (cutoff_utc, safe_batch),
            ).fetchall()
            job_ids = [str(row["job_id"]) for row in ids]
            deleted = 0
            if job_ids:
                placeholders = ",".join("?" for _ in job_ids)
                cur = conn.execute(
                    f"DELETE FROM jobs_archive WHERE job_id IN ({placeholders})",
                    tuple(job_ids),
                )
                deleted = int(cur.rowcount or 0)
            conn.execute("COMMIT")
        self._track_query(
            query_name="purge_archived_jobs",
            started_at=started_at,
            row_count=deleted,
            extra={"cutoff_utc": cutoff_utc, "batch_size": safe_batch},
        )
        return deleted

    def get_jobs(self, job_ids: list[str]) -> list[GuiJobRecord]:
        ids = [str(job_id) for job_id in job_ids if str(job_id).strip()]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE job_id IN ({placeholders}) ORDER BY created_at_utc DESC",
                tuple(ids),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def delete_jobs(self, job_ids: list[str]) -> int:
        ids = [str(job_id) for job_id in job_ids if str(job_id).strip()]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM job_logs WHERE job_id IN ({placeholders})",
                tuple(ids),
            )
            cur = conn.execute(
                f"DELETE FROM jobs WHERE job_id IN ({placeholders})",
                tuple(ids),
            )
            return int(cur.rowcount or 0)
