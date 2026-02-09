"""Stable runner API entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
from pydantic import ValidationError
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

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
from veldra.modeling import (
    build_study_name,
    run_tuning,
    train_binary_with_cv,
    train_multiclass_with_cv,
    train_regression_with_cv,
)

LOGGER = logging.getLogger("veldra")

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
_OPTUNA_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": optuna.logging.DEBUG,
    "INFO": optuna.logging.INFO,
    "WARNING": optuna.logging.WARNING,
    "ERROR": optuna.logging.ERROR,
}


def _ensure_config(config: RunConfig | dict[str, Any]) -> RunConfig:
    if isinstance(config, RunConfig):
        return config
    try:
        return RunConfig.model_validate(config)
    except ValidationError as exc:
        raise VeldraValidationError(f"Invalid RunConfig: {exc}") from exc


def fit(config: RunConfig | dict[str, Any]) -> RunResult:
    """Train supported model with CV and persist artifact payload."""
    parsed = _ensure_config(config)
    if parsed.task.type not in {"regression", "binary", "multiclass"}:
        raise VeldraNotImplementedError(
            "fit is currently implemented only for task.type='regression', 'binary', or "
            "'multiclass'."
        )
    if not parsed.data.path:
        raise VeldraValidationError("data.path is required for fit.")

    frame = load_tabular_data(parsed.data.path)
    if parsed.task.type == "regression":
        training_output = train_regression_with_cv(config=parsed, data=frame)
    elif parsed.task.type == "binary":
        training_output = train_binary_with_cv(config=parsed, data=frame)
    else:
        training_output = train_multiclass_with_cv(config=parsed, data=frame)

    run_id = uuid4().hex
    artifact_path = Path(parsed.export.artifact_dir) / run_id
    artifact = Artifact.from_config(
        run_config=parsed,
        run_id=run_id,
        feature_schema=training_output.feature_schema,
        model_text=training_output.model_text,
        metrics=training_output.metrics,
        cv_results=training_output.cv_results,
        calibrator=getattr(training_output, "calibrator", None),
        calibration_curve=getattr(training_output, "calibration_curve", None),
        threshold=getattr(training_output, "threshold", None),
        threshold_curve=getattr(training_output, "threshold_curve", None),
    )
    artifact.save(artifact_path)
    log_event(
        LOGGER,
        logging.INFO,
        "fit completed",
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
        metadata={
            "threshold_policy": (
                artifact.threshold.get("policy")
                if parsed.task.type == "binary" and artifact.threshold
                else None
            ),
            "threshold_value": (
                float(artifact.threshold.get("value"))
                if parsed.task.type == "binary" and artifact.threshold is not None
                else None
            ),
        },
    )


def tune(config: RunConfig | dict[str, Any]) -> TuneResult:
    parsed = _ensure_config(config)
    if parsed.task.type == "frontier":
        raise VeldraNotImplementedError(
            "tune is currently implemented only for task.type='regression', 'binary', or "
            "'multiclass'."
        )
    if parsed.task.type not in {"regression", "binary", "multiclass"}:
        raise VeldraValidationError(f"Unsupported task type for tune: '{parsed.task.type}'.")
    if not parsed.tuning.enabled:
        raise VeldraValidationError("tuning.enabled must be true to run tune().")
    if parsed.tuning.n_trials < 1:
        raise VeldraValidationError("tuning.n_trials must be >= 1.")
    if not parsed.data.path:
        raise VeldraValidationError("data.path is required for tune.")

    log_level = _LOG_LEVEL_MAP[parsed.tuning.log_level]
    optuna.logging.set_verbosity(_OPTUNA_LOG_LEVEL_MAP[parsed.tuning.log_level])

    frame = load_tabular_data(parsed.data.path)
    run_id = uuid4().hex
    study_name = build_study_name(parsed)
    tuning_path = Path(parsed.export.artifact_dir) / "tuning" / study_name
    tuning_path.mkdir(parents=True, exist_ok=True)
    storage_path = tuning_path / "study.db"
    storage_url = f"sqlite:///{storage_path.resolve()}"
    summary_path = tuning_path / "study_summary.json"
    trials_path = tuning_path / "trials.parquet"

    def _trial_progress(payload: dict[str, Any]) -> None:
        log_event(
            LOGGER,
            log_level,
            "tune trial completed",
            run_id=run_id,
            artifact_path=str(tuning_path),
            task_type=parsed.task.type,
            **payload,
        )

    tuning_output = run_tuning(
        config=parsed,
        data=frame,
        run_id=run_id,
        study_name=study_name,
        storage_url=storage_url,
        resume=parsed.tuning.resume,
        output_dir=tuning_path,
        on_trial_complete=_trial_progress,
    )

    log_event(
        LOGGER,
        log_level,
        "tune completed",
        run_id=run_id,
        artifact_path=str(tuning_path),
        task_type=parsed.task.type,
        n_trials=tuning_output.n_trials,
        metric_name=tuning_output.metric_name,
        study_name=study_name,
        resume=parsed.tuning.resume,
    )
    return TuneResult(
        run_id=run_id,
        task_type=parsed.task.type,
        best_params=tuning_output.best_params,
        best_score=tuning_output.best_score,
        metadata={
            "n_trials": tuning_output.n_trials,
            "metric_name": tuning_output.metric_name,
            "direction": tuning_output.direction,
            "seed": parsed.train.seed,
            "tuning_path": str(tuning_path),
            "summary_path": str(summary_path),
            "trials_path": str(trials_path),
            "study_name": study_name,
            "storage_url": storage_url,
            "resume": parsed.tuning.resume,
        },
    )


def evaluate(artifact_or_config: Any, data: Any) -> EvalResult:
    if not isinstance(artifact_or_config, Artifact):
        raise VeldraNotImplementedError(
            "evaluate currently supports Artifact input only."
        )

    artifact = artifact_or_config
    if artifact.run_config.task.type not in {"regression", "binary", "multiclass"}:
        raise VeldraNotImplementedError(
            "evaluate is currently implemented only for task.type='regression', 'binary', or "
            "'multiclass'."
        )
    if not isinstance(data, pd.DataFrame):
        raise VeldraValidationError("evaluate input must be a pandas.DataFrame.")
    if data.empty:
        raise VeldraValidationError("evaluate input DataFrame is empty.")

    target_col = artifact.run_config.data.target
    if target_col not in data.columns:
        raise VeldraValidationError(
            f"evaluate input is missing target column '{target_col}'."
        )

    y_true = data[target_col]
    x_eval = data.drop(columns=[target_col])

    if artifact.run_config.task.type == "regression":
        y_pred = artifact.predict(x_eval)
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    elif artifact.run_config.task.type == "binary":
        pred_frame = artifact.predict(x_eval)
        if not isinstance(pred_frame, pd.DataFrame):
            raise VeldraValidationError(
                "Binary artifact predict output must be a pandas.DataFrame."
            )
        if "p_cal" not in pred_frame.columns:
            raise VeldraValidationError("Binary predict output is missing required column 'p_cal'.")
        target_classes = artifact.feature_schema.get("target_classes")
        if not isinstance(target_classes, list) or len(target_classes) != 2:
            raise VeldraValidationError("feature_schema.target_classes is missing or invalid.")
        mapping = {target_classes[0]: 0, target_classes[1]: 1}
        y_encoded = y_true.map(mapping)
        if y_encoded.isna().any():
            raise VeldraValidationError(
                "evaluate input contains target values outside the artifact target classes."
            )
        y_binary = y_encoded.astype(int).to_numpy()
        if len(np.unique(y_binary)) < 2:
            raise VeldraValidationError(
                "Binary evaluation requires both classes to be present in input data."
            )
        p_cal = np.clip(pred_frame["p_cal"].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
        threshold_value = float((artifact.threshold or {}).get("value", 0.5))
        label_pred = (p_cal >= threshold_value).astype(int)
        metrics = {
            "auc": float(roc_auc_score(y_binary, p_cal)),
            "logloss": float(log_loss(y_binary, p_cal, labels=[0, 1])),
            "brier": float(brier_score_loss(y_binary, p_cal)),
            "accuracy": float(accuracy_score(y_binary, label_pred)),
            "f1": float(f1_score(y_binary, label_pred, zero_division=0)),
            "precision": float(precision_score(y_binary, label_pred, zero_division=0)),
            "recall": float(recall_score(y_binary, label_pred, zero_division=0)),
            "threshold": threshold_value,
        }
    else:
        pred_frame = artifact.predict(x_eval)
        if not isinstance(pred_frame, pd.DataFrame):
            raise VeldraValidationError(
                "Multiclass artifact predict output must be a pandas.DataFrame."
            )

        target_classes = artifact.feature_schema.get("target_classes")
        if not isinstance(target_classes, list) or len(target_classes) < 3:
            raise VeldraValidationError("feature_schema.target_classes is missing or invalid.")

        prob_cols = [f"proba_{label}" for label in target_classes]
        missing = [col for col in prob_cols if col not in pred_frame.columns]
        if missing:
            raise VeldraValidationError(
                f"Multiclass predict output is missing required columns: {missing}"
            )

        class_mapping = {label: idx for idx, label in enumerate(target_classes)}
        y_encoded = y_true.map(class_mapping)
        if y_encoded.isna().any():
            raise VeldraValidationError(
                "evaluate input contains target values outside the artifact target classes."
            )
        y_multiclass = y_encoded.astype(int).to_numpy()

        proba = np.asarray(pred_frame[prob_cols], dtype=float)
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        row_sum = proba.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise VeldraValidationError("Multiclass probability rows have invalid sums.")
        proba = proba / row_sum
        y_pred = np.argmax(proba, axis=1)

        metrics = {
            "accuracy": float(accuracy_score(y_multiclass, y_pred)),
            "macro_f1": float(f1_score(y_multiclass, y_pred, average="macro")),
            "logloss": float(
                log_loss(y_multiclass, proba, labels=list(range(len(target_classes))))
            ),
        }

    metadata = {
        "n_rows": int(len(data)),
        "target": target_col,
        "artifact_run_id": artifact.manifest.run_id,
        "threshold_policy": (
            artifact.threshold.get("policy")
            if artifact.run_config.task.type == "binary" and artifact.threshold
            else None
        ),
        "threshold_value": (
            float(artifact.threshold.get("value"))
            if artifact.run_config.task.type == "binary" and artifact.threshold is not None
            else None
        ),
    }

    log_event(
        LOGGER,
        logging.INFO,
        "evaluate completed",
        run_id=artifact.manifest.run_id,
        artifact_path=None,
        task_type=artifact.run_config.task.type,
        n_rows=metadata["n_rows"],
    )
    return EvalResult(
        task_type=artifact.run_config.task.type,
        metrics=metrics,
        metadata=metadata,
    )


def predict(artifact: Artifact, data: Any) -> Prediction:
    if artifact.run_config.task.type not in {"regression", "binary", "multiclass"}:
        raise VeldraNotImplementedError(
            "predict is currently implemented only for task.type='regression', 'binary', or "
            "'multiclass'."
        )
    if not isinstance(data, pd.DataFrame):
        raise VeldraValidationError("predict input must be a pandas.DataFrame.")
    pred = artifact.predict(data)
    n_rows = len(pred) if hasattr(pred, "__len__") else len(data)
    return Prediction(
        task_type=artifact.run_config.task.type,
        data=pred,
        metadata={"n_rows": int(n_rows)},
    )


def simulate(artifact: Artifact, data: Any, scenarios: Any) -> SimulationResult:
    _ = artifact, data, scenarios
    raise VeldraNotImplementedError("simulate is not implemented in MVP scaffold.")


def export(artifact: Artifact, format: str = "python") -> ExportResult:
    _ = artifact, format
    raise VeldraNotImplementedError("export is not implemented in MVP scaffold.")
