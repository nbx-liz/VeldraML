"""Public result types used by stable API functions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunResult:
    """Result payload returned by :func:`veldra.api.fit`.

    Attributes
    ----------
    run_id
        Unique identifier for the run.
    task_type
        Task type used for training.
    artifact_path
        Saved artifact directory path.
    metrics
        Aggregated metrics, usually mean CV metrics.
    metadata
        Additional contextual fields.
    """

    run_id: str
    task_type: str
    artifact_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TuneResult:
    """Result payload returned by :func:`veldra.api.tune`.

    Attributes
    ----------
    run_id
        Unique identifier for the tuning run.
    task_type
        Task type tuned.
    best_params
        Best hyperparameter assignment found by the study.
    best_score
        Best objective value.
    metadata
        Additional study metadata.
    """

    run_id: str
    task_type: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    """Result payload returned by :func:`veldra.api.evaluate`.

    Attributes
    ----------
    task_type
        Task type evaluated.
    metrics
        Task-specific evaluation metrics.
    metadata
        Evaluation context and auxiliary values.
    """

    task_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    """Result payload returned by :func:`veldra.api.predict`.

    Attributes
    ----------
    task_type
        Task type predicted.
    data
        Prediction output (array or dataframe by task).
    metadata
        Additional contextual values such as row counts.
    """

    task_type: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationResult:
    """Result payload returned by :func:`veldra.api.simulate`.

    Attributes
    ----------
    task_type
        Task type simulated.
    data
        Scenario comparison frame.
    metadata
        Simulation context and summary values.
    """

    task_type: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExportResult:
    """Result payload returned by :func:`veldra.api.export`.

    Attributes
    ----------
    path
        Export directory path.
    format
        Export format used.
    metadata
        Validation and artifact linkage metadata.
    """

    path: str
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CausalResult:
    """Result payload returned by :func:`veldra.api.estimate_dr`.

    Attributes
    ----------
    run_id
        Unique identifier for the causal run.
    method
        Causal method name (for example ``"dr"`` or ``"dr_did"``).
    estimand
        Target estimand (for example ``"att"`` or ``"ate"``).
    estimate
        Point estimate of the causal effect.
    std_error
        Standard error of the estimate when available.
    ci_lower
        Lower confidence interval bound when available.
    ci_upper
        Upper confidence interval bound when available.
    metrics
        Diagnostic and method-specific metrics.
    metadata
        Additional output locations and method context.
    """

    run_id: str
    method: str
    estimand: str
    estimate: float
    std_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
