"""Internal GUI dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GuiJobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
    "cancel_requested",
]


@dataclass(slots=True)
class ArtifactSummary:
    path: str
    run_id: str
    task_type: str
    created_at_utc: str | None = None


@dataclass(slots=True)
class RunInvocation:
    action: str
    config_yaml: str | None = None
    config_path: str | None = None
    data_path: str | None = None
    artifact_path: str | None = None
    scenarios_path: str | None = None
    export_format: str | None = None


@dataclass(slots=True)
class GuiRunResult:
    success: bool
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuiJobRecord:
    job_id: str
    status: GuiJobStatus
    action: str
    created_at_utc: str
    updated_at_utc: str
    invocation: RunInvocation
    cancel_requested: bool = False
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    result: GuiRunResult | None = None
    error_message: str | None = None


@dataclass(slots=True)
class GuiJobResult:
    job_id: str
    status: GuiJobStatus
    message: str
