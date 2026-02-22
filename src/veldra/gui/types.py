"""Internal GUI dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

GuiJobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
    "cancel_requested",
]
GuiJobPriority = Literal["low", "normal", "high"]
T = TypeVar("T")


@dataclass(slots=True)
class ArtifactSummary:
    path: str
    run_id: str
    task_type: str
    created_at_utc: str | None = None


@dataclass(slots=True)
class ArtifactSpec:
    artifact_path: str
    run_id: str
    task_type: str
    target_col: str
    feature_names: list[str]
    feature_dtypes: dict[str, str] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    created_at_utc: str | None = None


@dataclass(slots=True)
class PaginatedResult(Generic[T]):
    items: list[T]
    total_count: int
    limit: int
    offset: int


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = 0
    base_delay_sec: float = 1.0
    max_delay_sec: float = 8.0
    retry_on: tuple[str, ...] = ("timeout", "resource_busy", "io_transient")


@dataclass(slots=True)
class RunInvocation:
    action: str
    config_yaml: str | None = None
    config_path: str | None = None
    data_path: str | None = None
    artifact_path: str | None = None
    scenarios_path: str | None = None
    export_format: str | None = None
    priority: GuiJobPriority = "normal"
    retry_policy: RetryPolicy | None = None


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
    priority: GuiJobPriority = "normal"
    progress_pct: float = 0.0
    current_step: str | None = None
    cancel_requested: bool = False
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    result: GuiRunResult | None = None
    error_message: str | None = None
    retry_count: int = 0
    retry_parent_job_id: str | None = None
    last_error_kind: str | None = None


@dataclass(slots=True)
class GuiJobLogRecord:
    job_id: str
    seq: int
    created_at_utc: str
    level: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuiJobResult:
    job_id: str
    status: GuiJobStatus
    message: str
