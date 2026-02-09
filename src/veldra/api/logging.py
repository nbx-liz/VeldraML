"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any


def build_log_payload(
    run_id: str,
    artifact_path: str | None,
    task_type: str,
    **extra: Any,
) -> dict[str, Any]:
    """Return a payload with mandatory structured-log keys."""
    payload: dict[str, Any] = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "task_type": task_type,
    }
    payload.update(extra)
    return payload


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    run_id: str,
    artifact_path: str | None,
    task_type: str,
    **extra: Any,
) -> dict[str, Any]:
    """Emit a JSON log message with mandatory contract fields."""
    payload = build_log_payload(
        run_id=run_id,
        artifact_path=artifact_path,
        task_type=task_type,
        **extra,
    )
    logger.log(
        level,
        json.dumps(payload, sort_keys=True),
        extra={
            "run_id": payload["run_id"],
            "artifact_path": payload["artifact_path"],
            "task_type": payload["task_type"],
            "payload": payload,
            "event_message": message,
        },
    )
    return payload
