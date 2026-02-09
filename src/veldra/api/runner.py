"""Stable runner API entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import ValidationError

from veldra.api.artifact import Artifact
from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError
from veldra.api.logging import log_event
from veldra.api.types import (
    EvalResult,
    ExportResult,
    Prediction,
    RunResult,
    SimulationResult,
    TuneResult,
)
from veldra.config.models import RunConfig
from veldra.data import load_tabular_data
from veldra.modeling import train_regression_with_cv

LOGGER = logging.getLogger("veldra")


def _ensure_config(config: RunConfig | dict[str, Any]) -> RunConfig:
    if isinstance(config, RunConfig):
        return config
    try:
        return RunConfig.model_validate(config)
    except ValidationError as exc:
        raise VeldraValidationError(f"Invalid RunConfig: {exc}") from exc


def fit(config: RunConfig | dict[str, Any]) -> RunResult:
    """Train regression model with CV and persist artifact payload."""
    parsed = _ensure_config(config)
    if parsed.task.type != "regression":
        raise VeldraNotImplementedError(
            "fit is currently implemented only for task.type='regression'."
        )
    if not parsed.data.path:
        raise VeldraValidationError("data.path is required for fit.")

    frame = load_tabular_data(parsed.data.path)
    training_output = train_regression_with_cv(config=parsed, data=frame)

    run_id = uuid4().hex
    artifact_path = Path(parsed.export.artifact_dir) / run_id
    artifact = Artifact.from_config(
        run_config=parsed,
        run_id=run_id,
        feature_schema=training_output.feature_schema,
        model_text=training_output.model_text,
        metrics=training_output.metrics,
        cv_results=training_output.cv_results,
    )
    artifact.save(artifact_path)
    log_event(
        LOGGER,
        logging.INFO,
        "fit scaffold completed",
        run_id=run_id,
        artifact_path=str(artifact_path),
        task_type=parsed.task.type,
    )
    mean_metrics = training_output.metrics.get("mean", {})
    return RunResult(
        run_id=run_id,
        task_type=parsed.task.type,
        artifact_path=str(artifact_path),
        metrics={k: float(v) for k, v in mean_metrics.items()},
    )


def tune(config: RunConfig | dict[str, Any]) -> TuneResult:
    _ = _ensure_config(config)
    raise VeldraNotImplementedError("tune is not implemented in MVP scaffold.")


def evaluate(artifact_or_config: Any, data: Any) -> EvalResult:
    _ = artifact_or_config, data
    raise VeldraNotImplementedError("evaluate is not implemented in MVP scaffold.")


def predict(artifact: Artifact, data: Any) -> Prediction:
    if artifact.run_config.task.type != "regression":
        raise VeldraNotImplementedError(
            "predict is currently implemented only for task.type='regression'."
        )
    if not isinstance(data, pd.DataFrame):
        raise VeldraValidationError("predict input must be a pandas.DataFrame.")
    pred = artifact.predict(data)
    return Prediction(
        task_type=artifact.run_config.task.type,
        data=pred,
        metadata={"n_rows": int(len(pred))},
    )


def simulate(artifact: Artifact, data: Any, scenarios: Any) -> SimulationResult:
    _ = artifact, data, scenarios
    raise VeldraNotImplementedError("simulate is not implemented in MVP scaffold.")


def export(artifact: Artifact, format: str = "python") -> ExportResult:
    _ = artifact, format
    raise VeldraNotImplementedError("export is not implemented in MVP scaffold.")
