"""Stable runner API entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

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

LOGGER = logging.getLogger("veldra")


def _ensure_config(config: RunConfig | dict[str, Any]) -> RunConfig:
    if isinstance(config, RunConfig):
        return config
    try:
        return RunConfig.model_validate(config)
    except ValidationError as exc:
        raise VeldraValidationError(f"Invalid RunConfig: {exc}") from exc


def fit(config: RunConfig | dict[str, Any]) -> RunResult:
    """MVP fit scaffold that validates config and writes reproducible artifact metadata."""
    parsed = _ensure_config(config)
    run_id = uuid4().hex
    artifact_path = Path(parsed.export.artifact_dir) / run_id
    artifact = Artifact.from_config(run_config=parsed, run_id=run_id, feature_schema={})
    artifact.save(artifact_path)
    log_event(
        LOGGER,
        logging.INFO,
        "fit scaffold completed",
        run_id=run_id,
        artifact_path=str(artifact_path),
        task_type=parsed.task.type,
    )
    return RunResult(run_id=run_id, task_type=parsed.task.type, artifact_path=str(artifact_path))


def tune(config: RunConfig | dict[str, Any]) -> TuneResult:
    _ = _ensure_config(config)
    raise VeldraNotImplementedError("tune is not implemented in MVP scaffold.")


def evaluate(artifact_or_config: Any, data: Any) -> EvalResult:
    _ = artifact_or_config, data
    raise VeldraNotImplementedError("evaluate is not implemented in MVP scaffold.")


def predict(artifact: Artifact, data: Any) -> Prediction:
    _ = artifact, data
    raise VeldraNotImplementedError("predict is not implemented in MVP scaffold.")


def simulate(artifact: Artifact, data: Any, scenarios: Any) -> SimulationResult:
    _ = artifact, data, scenarios
    raise VeldraNotImplementedError("simulate is not implemented in MVP scaffold.")


def export(artifact: Artifact, format: str = "python") -> ExportResult:
    _ = artifact, format
    raise VeldraNotImplementedError("export is not implemented in MVP scaffold.")
