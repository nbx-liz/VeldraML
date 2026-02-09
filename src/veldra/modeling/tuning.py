"""Hyperparameter tuning routines built on top of existing CV trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling.binary import train_binary_with_cv
from veldra.modeling.multiclass import train_multiclass_with_cv
from veldra.modeling.regression import train_regression_with_cv


@dataclass(slots=True)
class TuningOutput:
    best_params: dict[str, Any]
    best_score: float
    metric_name: str
    direction: str
    n_trials: int
    trials: pd.DataFrame


def _objective_spec(task_type: str) -> tuple[str, str]:
    mapping = {
        "regression": ("rmse", "minimize"),
        "binary": ("auc", "maximize"),
        "multiclass": ("macro_f1", "maximize"),
    }
    if task_type not in mapping:
        raise VeldraValidationError(f"Unsupported task type for tuning: '{task_type}'.")
    return mapping[task_type]


def _default_search_space(task_type: str, preset: str) -> dict[str, Any]:
    if preset == "fast":
        return {
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.2, "log": True},
            "num_leaves": {"type": "int", "low": 16, "high": 64},
            "feature_fraction": {"type": "float", "low": 0.7, "high": 1.0},
        }
    if preset == "standard":
        return {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 16, "high": 128},
            "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            "min_data_in_leaf": {"type": "int", "low": 10, "high": 120},
            "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            "bagging_freq": {"type": "int", "low": 1, "high": 7},
        }
    raise VeldraValidationError(
        f"Unsupported tuning preset '{preset}' for task '{task_type}'."
    )


def _resolve_search_space(config: RunConfig) -> dict[str, Any]:
    if config.tuning.search_space:
        return config.tuning.search_space
    return _default_search_space(config.task.type, config.tuning.preset)


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: Any) -> Any:
    if isinstance(spec, dict):
        param_type = spec.get("type")
        if param_type == "int":
            return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        if param_type == "float":
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        if param_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise VeldraValidationError(
                    f"search_space.{name}.choices must be a non-empty list."
                )
            return trial.suggest_categorical(name, choices)
        raise VeldraValidationError(
            f"search_space.{name}.type must be one of int/float/categorical."
        )
    if isinstance(spec, list):
        if not spec:
            raise VeldraValidationError(f"search_space.{name} list must not be empty.")
        return trial.suggest_categorical(name, spec)
    return spec


def _build_trial_config(config: RunConfig, trial_params: dict[str, Any]) -> RunConfig:
    trial_cfg = config.model_copy(deep=True)
    trial_cfg.train.lgb_params = {
        **trial_cfg.train.lgb_params,
        **trial_params,
    }
    if (
        trial_cfg.task.type == "binary"
        and trial_cfg.postprocess.threshold_optimization is not None
    ):
        trial_cfg.postprocess.threshold_optimization.enabled = False
    return trial_cfg


def _score_for_task(config: RunConfig, data: pd.DataFrame, metric_name: str) -> float:
    if config.task.type == "regression":
        output = train_regression_with_cv(config=config, data=data)
    elif config.task.type == "binary":
        output = train_binary_with_cv(config=config, data=data)
    elif config.task.type == "multiclass":
        output = train_multiclass_with_cv(config=config, data=data)
    else:
        raise VeldraValidationError(f"Unsupported tuning task type '{config.task.type}'.")

    mean_metrics = output.metrics.get("mean", {})
    if metric_name not in mean_metrics:
        raise VeldraValidationError(
            f"Tuning metric '{metric_name}' is missing from training output."
        )
    return float(mean_metrics[metric_name])


def _study_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": int(trial.number),
            "value": None if trial.value is None else float(trial.value),
            "state": str(trial.state.name),
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        records.append(row)
    return pd.DataFrame.from_records(records)


def run_tuning(config: RunConfig, data: pd.DataFrame) -> TuningOutput:
    """Run Optuna tuning and return summary payload for API adapter layer."""
    if config.task.type not in {"regression", "binary", "multiclass"}:
        raise VeldraValidationError(
            "run_tuning supports only regression/binary/multiclass tasks."
        )

    metric_name, direction = _objective_spec(config.task.type)
    search_space = _resolve_search_space(config)

    sampler = optuna.samplers.TPESampler(seed=config.train.seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for name, spec in search_space.items():
            params[name] = _suggest_from_spec(trial, name, spec)
        trial_cfg = _build_trial_config(config, params)
        return _score_for_task(trial_cfg, data, metric_name)

    study.optimize(objective, n_trials=config.tuning.n_trials)

    trials = _study_trials_dataframe(study)
    best = study.best_trial
    return TuningOutput(
        best_params=dict(best.params),
        best_score=float(best.value),
        metric_name=metric_name,
        direction=direction,
        n_trials=int(len(study.trials)),
        trials=trials,
    )

