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
    """Build structured log payload with mandatory keys.

    Parameters
    ----------
    run_id
        Run identifier.
    artifact_path
        Artifact path linked to the event, if available.
    task_type
        Task type of the event.
    **extra
        Additional fields to include.

    Returns
    -------
    dict[str, Any]
        JSON-serializable payload with mandatory and extra fields.
    """
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
    """Emit one structured log event.

    Parameters
    ----------
    logger
        Target logger instance.
    level
        Logging level.
    message
        Human-readable event message.
    run_id
        Run identifier.
    artifact_path
        Artifact path linked to the event, if available.
    task_type
        Task type of the event.
    **extra
        Additional payload fields.

    Returns
    -------
    dict[str, Any]
        Payload that was emitted.
    """
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
