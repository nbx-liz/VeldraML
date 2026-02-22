"""Shared CV runner for modeling tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling.utils import split_for_early_stopping as _default_split_for_early_stopping

FoldMetrics = dict[str, float]
FoldRecord = dict[str, float | int]
FitBoosterFn = Callable[
    [
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        RunConfig,
        dict[str, Any],
        dict[str, Any],
    ],
    Any,
]


@dataclass(slots=True)
class TaskSpec:
    fit_booster: FitBoosterFn
    predict_valid: Callable[[Any, pd.DataFrame, dict[str, Any]], np.ndarray]
    fold_metrics: Callable[[pd.Series, np.ndarray, dict[str, Any]], FoldMetrics]
    fold_record: Callable[[int, FoldMetrics, int, int, dict[str, Any]], FoldRecord]
    mean_metrics: Callable[[pd.Series, np.ndarray, dict[str, Any]], FoldMetrics]
    observation_table: Callable[[np.ndarray, pd.Series, np.ndarray, dict[str, Any]], pd.DataFrame]
    init_oof: Callable[[int, dict[str, Any]], np.ndarray]


@dataclass(slots=True)
class CVRunnerResult:
    oof: np.ndarray
    oof_valid_mask: np.ndarray
    fold_ids: np.ndarray
    fold_records: list[FoldRecord]
    cv_results: pd.DataFrame
    mean_metrics: FoldMetrics
    final_model: Any
    training_history: dict[str, Any]
    observation_table: pd.DataFrame


def booster_iteration_stats(booster: Any, fallback_rounds: int) -> tuple[int, int]:
    """Resolve best/current iteration from LightGBM booster-like objects."""
    current_iteration_fn = getattr(booster, "current_iteration", None)
    current_iteration = (
        int(current_iteration_fn()) if callable(current_iteration_fn) else int(fallback_rounds)
    )
    best_iteration = int(getattr(booster, "best_iteration", 0) or current_iteration)
    if best_iteration <= 0:
        best_iteration = current_iteration
    return best_iteration, current_iteration


def run_cv_training(
    *,
    config: RunConfig,
    x: pd.DataFrame,
    y: pd.Series,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    spec: TaskSpec,
    context: dict[str, Any] | None = None,
    split_for_early_stopping_fn: Callable[
        [pd.DataFrame, pd.Series, RunConfig],
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ] = _default_split_for_early_stopping,
) -> CVRunnerResult:
    """Run common CV loop and final-model training across tasks."""
    ctx = {} if context is None else context
    oof = spec.init_oof(len(x), ctx)
    fold_ids = np.full(len(x), -1, dtype=int)
    fold_records: list[FoldRecord] = []
    history_folds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise VeldraValidationError("Encountered an empty train/valid split.")
        x_fold_train = x.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        x_es_train, x_es_valid, y_es_train, y_es_valid = split_for_early_stopping_fn(
            x_fold_train, y_fold_train, config
        )
        eval_history: dict[str, Any] = {}
        booster = spec.fit_booster(
            x_es_train,
            y_es_train,
            x_es_valid,
            y_es_valid,
            config,
            ctx,
            eval_history,
        )
        pred = np.asarray(spec.predict_valid(booster, x.iloc[valid_idx], ctx), dtype=float)
        if oof.ndim == 1:
            oof[valid_idx] = pred
        else:
            oof[valid_idx, :] = pred
        fold_ids[valid_idx] = fold_idx

        best_iteration, num_iterations = booster_iteration_stats(
            booster, config.train.num_boost_round
        )
        history_folds.append(
            {
                "fold": fold_idx,
                "best_iteration": best_iteration if best_iteration > 0 else num_iterations,
                "num_iterations": num_iterations,
                "eval_history": eval_history.get("valid_0", eval_history),
            }
        )
        metrics = spec.fold_metrics(y.iloc[valid_idx], pred, ctx)
        fold_records.append(
            spec.fold_record(
                fold_idx,
                metrics,
                int(len(train_idx)),
                int(len(valid_idx)),
                ctx,
            )
        )

    if oof.ndim == 1:
        oof_valid_mask = ~np.isnan(oof)
    else:
        oof_valid_mask = ~np.isnan(oof).any(axis=1)

    if config.split.type != "timeseries":
        if not bool(np.all(oof_valid_mask)):
            raise VeldraValidationError(
                "OOF predictions contain missing values. Check split configuration."
            )
    elif not bool(np.any(oof_valid_mask)):
        raise VeldraValidationError(
            "Timeseries split produced no scored OOF rows. Check split configuration."
        )

    x_final_train, x_final_valid, y_final_train, y_final_valid = split_for_early_stopping_fn(
        x, y, config
    )
    final_eval_history: dict[str, Any] = {}
    final_model = spec.fit_booster(
        x_final_train,
        y_final_train,
        x_final_valid,
        y_final_valid,
        config,
        ctx,
        final_eval_history,
    )
    final_best_iteration, final_num_iterations = booster_iteration_stats(
        final_model, config.train.num_boost_round
    )
    training_history = {
        "folds": history_folds,
        "final_model": {
            "best_iteration": (
                final_best_iteration if final_best_iteration > 0 else final_num_iterations
            ),
            "num_iterations": final_num_iterations,
            "eval_history": final_eval_history.get("valid_0", final_eval_history),
        },
        "oof_total_rows": int(len(y)),
        "oof_scored_rows": int(np.sum(oof_valid_mask)),
        "oof_coverage_ratio": float(np.mean(oof_valid_mask.astype(float))),
    }

    y_for_metrics = y.iloc[oof_valid_mask]
    oof_for_metrics = oof[oof_valid_mask] if oof.ndim == 1 else oof[oof_valid_mask, :]

    return CVRunnerResult(
        oof=oof,
        oof_valid_mask=oof_valid_mask,
        fold_ids=fold_ids,
        fold_records=fold_records,
        cv_results=pd.DataFrame.from_records(fold_records),
        mean_metrics=spec.mean_metrics(y_for_metrics, oof_for_metrics, ctx),
        final_model=final_model,
        training_history=training_history,
        observation_table=spec.observation_table(fold_ids, y, oof, ctx),
    )
