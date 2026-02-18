"""SQLite-backed job persistence for GUI async runs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veldra.gui.types import GuiJobPriority, GuiJobRecord, GuiRunResult, RunInvocation

TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
_PRIORITY_TO_INT: dict[GuiJobPriority, int] = {"low": 10, "normal": 50, "high": 90}


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
    return RunInvocation(**payload)


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
        cancel_requested=bool(int(row["cancel_requested"])),
        started_at_utc=(str(row["started_at_utc"]) if row["started_at_utc"] is not None else None),
        finished_at_utc=(
            str(row["finished_at_utc"]) if row["finished_at_utc"] is not None else None
        ),
        result=_decode_result(row["result_json"]),
        error_message=(str(row["error_message"]) if row["error_message"] is not None else None),
    )


class GuiJobStore:
    """Persist GUI run jobs in SQLite."""

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
                    started_at_utc TEXT NULL,
                    finished_at_utc TEXT NULL,
                    result_json TEXT NULL,
                    error_message TEXT NULL
                )
                """
            )
            columns = conn.execute("PRAGMA table_info(jobs)").fetchall()
            column_names = {str(row["name"]) for row in columns}
            if "priority" not in column_names:
                conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 50")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at_utc)"
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status_priority_created
                ON jobs(status, priority DESC, created_at_utc ASC)
                """
            )

    def enqueue_job(self, invocation: RunInvocation) -> GuiJobRecord:
        now = _utcnow_iso()
        job_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, status, action, priority, created_at_utc, updated_at_utc,
                    invocation_json, cancel_requested, started_at_utc, finished_at_utc,
                    result_json, error_message
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL)
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
    ) -> GuiJobRecord | None:
        now = _utcnow_iso()
        result = GuiRunResult(success=False, message=message, payload=payload or {})
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    result_json = ?,
                    error_message = ?
                WHERE job_id = ?
                """,
                (now, now, json.dumps(asdict(result), ensure_ascii=False), message, job_id),
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
                    updated_at_utc = ?,
                    finished_at_utc = ?,
                    result_json = ?,
                    error_message = NULL
                WHERE job_id = ?
                """,
                (now, now, json.dumps(asdict(result), ensure_ascii=False), job_id),
            )
        return self.get_job(job_id)

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
        safe_limit = max(1, min(int(limit), 1000))
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at_utc DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at_utc DESC LIMIT ?",
                    (status, safe_limit),
                ).fetchall()
        return [_row_to_record(row) for row in rows]

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
            cur = conn.execute(
                f"DELETE FROM jobs WHERE job_id IN ({placeholders})",
                tuple(ids),
            )
            return int(cur.rowcount or 0)
