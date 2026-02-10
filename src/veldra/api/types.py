"""Public result types used by stable API functions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunResult:
    run_id: str
    task_type: str
    artifact_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TuneResult:
    run_id: str
    task_type: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    task_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    task_type: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationResult:
    task_type: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExportResult:
    path: str
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CausalResult:
    run_id: str
    method: str
    estimand: str
    estimate: float
    std_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
