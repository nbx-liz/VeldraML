"""Hyperparameter tuning routines built on top of existing CV trainers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import optuna
import pandas as pd
from optuna.exceptions import DuplicatedStudyError

from veldra.api.exceptions import VeldraValidationError
from veldra.causal import run_dr_did_estimation, run_dr_estimation
from veldra.config.models import RunConfig, resolve_tuning_objective
from veldra.modeling.binary import train_binary_with_cv
from veldra.modeling.frontier import train_frontier_with_cv
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
    study_name: str
    storage_url: str
    best_components: dict[str, Any]


def _objective_spec(
    task_type: str,
    objective: str | None,
    *,
    causal_method: str | None = None,
) -> tuple[str, str]:
    try:
        metric_name = resolve_tuning_objective(
            task_type,
            objective,
            causal_method=causal_method,
        )
    except ValueError as exc:
        raise VeldraValidationError(str(exc)) from exc
    direction_by_metric = {
        "rmse": "minimize",
        "mae": "minimize",
        "r2": "maximize",
        "pinball": "minimize",
        "pinball_coverage_penalty": "minimize",
        "auc": "maximize",
        "logloss": "minimize",
        "brier": "minimize",
        "accuracy": "maximize",
        "f1": "maximize",
        "precision": "maximize",
        "recall": "maximize",
        "macro_f1": "maximize",
        "dr_std_error": "minimize",
        "dr_overlap_penalty": "minimize",
        "drdid_std_error": "minimize",
        "drdid_overlap_penalty": "minimize",
    }
    if metric_name not in direction_by_metric:
        raise VeldraValidationError(f"Unsupported objective metric '{metric_name}'.")
    return metric_name, direction_by_metric[metric_name]


def build_study_name(config: RunConfig) -> str:
    if config.tuning.study_name:
        return config.tuning.study_name
    payload = config.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{config.task.type}_tune_{suffix}"


def _default_search_space(task_type: str, preset: str) -> dict[str, Any]:
    _ = task_type
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


def _resolve_frontier_tuning_terms(config: RunConfig) -> tuple[float, float, float]:
    coverage_target = (
        float(config.frontier.alpha)
        if config.tuning.coverage_target is None
        else float(config.tuning.coverage_target)
    )
    coverage_tolerance = float(config.tuning.coverage_tolerance)
    penalty_weight = float(config.tuning.penalty_weight)
    return coverage_target, coverage_tolerance, penalty_weight


def _frontier_objective_from_metrics(
    config: RunConfig,
    metric_name: str,
    mean_metrics: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    if "pinball" not in mean_metrics:
        raise VeldraValidationError("Tuning metric 'pinball' is missing from training output.")
    pinball = float(mean_metrics["pinball"])
    coverage = (
        float(mean_metrics["coverage"])
        if "coverage" in mean_metrics
        else None
    )
    coverage_target, coverage_tolerance, penalty_weight = _resolve_frontier_tuning_terms(config)
    if metric_name == "pinball_coverage_penalty" and coverage is None:
        raise VeldraValidationError(
            "Tuning metric 'coverage' is required for objective 'pinball_coverage_penalty'."
        )
    coverage_gap = abs(coverage - coverage_target) if coverage is not None else 0.0
    penalty = penalty_weight * max(0.0, coverage_gap - coverage_tolerance)
    objective_value = pinball
    if metric_name == "pinball_coverage_penalty":
        objective_value = pinball + penalty

    components = {
        "pinball": pinball,
        "coverage": coverage,
        "coverage_target": coverage_target,
        "coverage_tolerance": coverage_tolerance,
        "coverage_gap": coverage_gap,
        "penalty_weight": penalty_weight,
        "penalty": penalty,
        "objective_value": objective_value,
    }
    return float(objective_value), components


def _score_for_task_with_components(
    config: RunConfig,
    data: pd.DataFrame,
    metric_name: str,
) -> tuple[float, dict[str, Any]]:
    if config.causal is not None:
        if config.causal.method == "dr":
            estimation = run_dr_estimation(config, data)
        elif config.causal.method == "dr_did":
            estimation = run_dr_did_estimation(config, data)
        else:
            raise VeldraValidationError(
                f"Unsupported causal method '{config.causal.method}' for tuning."
            )
        std_error = (
            float(estimation.std_error)
            if estimation.std_error is not None
            else float("inf")
        )
        overlap_metric = float(estimation.metrics.get("overlap_metric", 0.0))
        penalty_weight = float(config.tuning.causal_penalty_weight)
        penalty = 0.0
        if metric_name in {"dr_overlap_penalty", "drdid_overlap_penalty"}:
            penalty = penalty_weight * max(0.0, 0.1 - overlap_metric)
        objective_value = std_error + penalty
        components = {
            "estimate": float(estimation.estimate),
            "std_error": std_error,
            "overlap_metric": overlap_metric,
            "penalty_weight": penalty_weight,
            "penalty": penalty,
            "objective_value": objective_value,
        }
        return objective_value, components

    if config.task.type == "regression":
        output = train_regression_with_cv(config=config, data=data)
    elif config.task.type == "binary":
        output = train_binary_with_cv(config=config, data=data)
    elif config.task.type == "multiclass":
        output = train_multiclass_with_cv(config=config, data=data)
    elif config.task.type == "frontier":
        output = train_frontier_with_cv(config=config, data=data)
    else:
        raise VeldraValidationError(f"Unsupported tuning task type '{config.task.type}'.")

    mean_metrics = output.metrics.get("mean", {})
    if (
        config.task.type == "frontier"
        and metric_name in {"pinball", "pinball_coverage_penalty"}
    ):
        return _frontier_objective_from_metrics(config, metric_name, mean_metrics)

    if metric_name not in mean_metrics:
        raise VeldraValidationError(
            f"Tuning metric '{metric_name}' is missing from training output."
        )
    objective_value = float(mean_metrics[metric_name])
    return objective_value, {"objective_value": objective_value}


def _score_for_task(config: RunConfig, data: pd.DataFrame, metric_name: str) -> float:
    objective_value, _ = _score_for_task_with_components(config, data, metric_name)
    return objective_value


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
        for key, value in trial.user_attrs.items():
            row[key] = value
        records.append(row)
    return pd.DataFrame.from_records(records)


def _best_snapshot(study: optuna.Study) -> tuple[float | None, dict[str, Any], dict[str, Any]]:
    try:
        trial = study.best_trial
    except ValueError:
        return None, {}, {}
    if trial.value is None:
        return None, {}, {}
    components = {
        str(key): value
        for key, value in trial.user_attrs.items()
        if isinstance(value, (int, float, bool, str))
    }
    return float(trial.value), dict(trial.params), components


def _write_tuning_artifacts(
    output_dir: Path,
    summary: dict[str, Any],
    trials: pd.DataFrame,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "study_summary.json"
    trials_path = output_dir / "trials.parquet"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    trials.to_parquet(trials_path, index=False)
    return summary_path, trials_path


def run_tuning(
    config: RunConfig,
    data: pd.DataFrame,
    *,
    run_id: str,
    study_name: str,
    storage_url: str,
    resume: bool,
    output_dir: Path,
    on_trial_complete: Callable[[dict[str, Any]], None] | None = None,
) -> TuningOutput:
    """Run Optuna tuning and return summary payload for API adapter layer."""
    if config.task.type not in {"regression", "binary", "multiclass", "frontier"}:
        raise VeldraValidationError(
            "run_tuning supports only regression/binary/multiclass/frontier tasks."
        )

    metric_name, direction = _objective_spec(
        config.task.type,
        config.tuning.objective,
        causal_method=config.causal.method if config.causal is not None else None,
    )
    search_space = _resolve_search_space(config)
    sampler = optuna.samplers.TPESampler(seed=config.train.seed)

    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            storage=storage_url,
            load_if_exists=resume,
        )
    except DuplicatedStudyError as exc:
        raise VeldraValidationError(
            f"Study '{study_name}' already exists. Set tuning.resume=true to continue or use "
            "a different tuning.study_name."
        ) from exc

    def callback(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best_value, best_params, best_components = _best_snapshot(study_obj)
        trials_df = _study_trials_dataframe(study_obj)
        summary = {
            "run_id": run_id,
            "task_type": config.task.type,
            "study_name": study_name,
            "storage_url": storage_url,
            "metric_name": metric_name,
            "direction": direction,
            "best_score": best_value,
            "best_params": best_params,
            "objective_components": best_components,
            "n_trials": int(len(study_obj.trials)),
            "seed": config.train.seed,
        }
        _write_tuning_artifacts(output_dir, summary, trials_df)
        if on_trial_complete is not None:
            payload = {
                "trial_number": int(trial.number),
                "trial_value": None if trial.value is None else float(trial.value),
                "best_value": best_value,
                "n_trials_done": int(len(study_obj.trials)),
            }
            payload.update(
                {
                    str(key): value
                    for key, value in trial.user_attrs.items()
                    if isinstance(value, (int, float, bool, str))
                }
            )
            on_trial_complete(
                payload
            )

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for name, spec in search_space.items():
            params[name] = _suggest_from_spec(trial, name, spec)
        trial_cfg = _build_trial_config(config, params)
        objective_value, components = _score_for_task_with_components(trial_cfg, data, metric_name)
        for key, value in components.items():
            trial.set_user_attr(key, value)
        return objective_value

    study.optimize(objective, n_trials=config.tuning.n_trials, callbacks=[callback])

    best_value, best_params, best_components = _best_snapshot(study)
    if best_value is None:
        raise VeldraValidationError("Tuning finished without a completed trial.")

    trials = _study_trials_dataframe(study)
    summary = {
        "run_id": run_id,
        "task_type": config.task.type,
        "study_name": study_name,
        "storage_url": storage_url,
        "metric_name": metric_name,
        "direction": direction,
        "best_score": best_value,
        "best_params": best_params,
        "objective_components": best_components,
        "n_trials": int(len(study.trials)),
        "seed": config.train.seed,
    }
    _write_tuning_artifacts(output_dir, summary, trials)
    return TuningOutput(
        best_params=best_params,
        best_score=best_value,
        metric_name=metric_name,
        direction=direction,
        n_trials=int(len(study.trials)),
        trials=trials,
        study_name=study_name,
        storage_url=storage_url,
        best_components=best_components,
    )
