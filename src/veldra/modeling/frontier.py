"""Frontier-style quantile regression training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling.utils import (
    resolve_auto_num_leaves,
    resolve_feature_weights,
    resolve_ratio_params,
    split_for_early_stopping,
)
from veldra.split import iter_cv_splits


@dataclass(slots=True)
class FrontierTrainingOutput:
    model_text: str
    metrics: dict[str, Any]
    cv_results: pd.DataFrame
    feature_schema: dict[str, Any]
    training_history: dict[str, Any]


def _booster_iteration_stats(booster: Any, fallback_rounds: int) -> tuple[int, int]:
    current_iteration_fn = getattr(booster, "current_iteration", None)
    current_iteration = (
        int(current_iteration_fn()) if callable(current_iteration_fn) else int(fallback_rounds)
    )
    best_iteration = int(getattr(booster, "best_iteration", 0) or current_iteration)
    if best_iteration <= 0:
        best_iteration = current_iteration
    return best_iteration, current_iteration


def _build_feature_frame(config: RunConfig, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = config.data.target
    if target not in data.columns:
        raise VeldraValidationError(f"Target column '{target}' was not found in input data.")
    if data.empty:
        raise VeldraValidationError("Input data is empty.")

    y = data[target].copy()
    if y.isna().any():
        raise VeldraValidationError("Target column contains null values.")

    drop_cols = set(config.data.drop_cols + config.data.id_cols + [target])
    feature_cols = [col for col in data.columns if col not in drop_cols]
    if not feature_cols:
        raise VeldraValidationError(
            "No training features remain after applying drop/id/target columns."
        )

    x = data.loc[:, feature_cols].copy()
    return x, y


def _train_single_booster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: RunConfig,
    evaluation_history: dict[str, Any] | None = None,
) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": float(config.frontier.alpha),
        "metric": "quantile",
        "verbosity": -1,
        "seed": config.train.seed,
        **config.train.lgb_params,
    }
    resolved_leaves = resolve_auto_num_leaves(config)
    if resolved_leaves is not None:
        params["num_leaves"] = resolved_leaves
    params.update(resolve_ratio_params(config, len(x_train)))
    feature_weights = resolve_feature_weights(config, list(x_train.columns))
    if feature_weights is not None:
        params["feature_weights"] = feature_weights
        params["feature_pre_filter"] = False

    categorical = [col for col in config.data.categorical if col in x_train.columns]
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        categorical_feature=[col for col in categorical if col in x_valid.columns],
        free_raw_data=False,
    )
    callbacks = []
    if config.train.early_stopping_rounds:
        callbacks.append(lgb.early_stopping(config.train.early_stopping_rounds, verbose=False))
    if evaluation_history is not None:
        callbacks.append(lgb.record_evaluation(evaluation_history))

    return lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[valid_set],
        num_boost_round=config.train.num_boost_round,
        callbacks=callbacks,
    )


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    error = y_true - y_pred
    loss = np.maximum(alpha * error, (alpha - 1.0) * error)
    return float(np.mean(loss))


def _frontier_metrics(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> dict[str, float]:
    y_true_f = np.asarray(y_true, dtype=float)
    y_pred_f = np.asarray(y_pred, dtype=float)
    u_hat = np.maximum(0.0, y_pred_f - y_true_f)
    return {
        "pinball": _pinball_loss(y_true_f, y_pred_f, alpha),
        "mae": float(mean_absolute_error(y_true_f, y_pred_f)),
        "mean_u_hat": float(np.mean(u_hat)),
        "coverage": float(np.mean(y_true_f <= y_pred_f)),
    }


def train_frontier_with_cv(config: RunConfig, data: pd.DataFrame) -> FrontierTrainingOutput:
    """Train frontier quantile model with CV and return artifact payload.

    Notes
    -----
    - The model uses LightGBM quantile objective with ``frontier.alpha``.
    - Fold predictions are aggregated as OOF predictions to compute pinball,
      MAE, mean inefficiency (``u_hat``), and empirical coverage.
    - Timeseries mode preserves chronological order before split generation.
    """
    if config.task.type != "frontier":
        raise VeldraValidationError("train_frontier_with_cv only supports task.type='frontier'.")
    if not config.data.path:
        raise VeldraValidationError("data.path is required for fit.")

    if config.split.type == "timeseries":
        data = data.sort_values(config.split.time_col).reset_index(drop=True)

    x, y = _build_feature_frame(config, data)
    if config.split.type == "stratified":
        raise VeldraValidationError("split.type='stratified' is not supported for frontier task.")
    splits = iter_cv_splits(config, data, x)
    alpha = float(config.frontier.alpha)

    oof_pred = np.full(len(x), np.nan, dtype=float)
    fold_records: list[dict[str, float | int]] = []
    history_folds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise VeldraValidationError("Encountered an empty train/valid split.")
        x_fold_train = x.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        x_es_train, x_es_valid, y_es_train, y_es_valid = split_for_early_stopping(
            x_fold_train, y_fold_train, config
        )
        eval_history: dict[str, Any] = {}

        booster = _train_single_booster(
            x_train=x_es_train,
            y_train=y_es_train,
            x_valid=x_es_valid,
            y_valid=y_es_valid,
            config=config,
            evaluation_history=eval_history,
        )
        pred = np.asarray(
            booster.predict(x.iloc[valid_idx], num_iteration=booster.best_iteration),
            dtype=float,
        )
        if pred.ndim == 0:
            pred = np.full(len(valid_idx), float(pred))
        if pred.size == 1 and len(valid_idx) > 1:
            pred = np.full(len(valid_idx), float(pred.item()))
        if pred.shape[0] != len(valid_idx):
            raise VeldraValidationError(
                "Frontier prediction output length does not match validation rows."
            )
        oof_pred[valid_idx] = pred

        fold_metrics = _frontier_metrics(y.iloc[valid_idx].to_numpy(), pred, alpha)
        best_iteration, num_iterations = _booster_iteration_stats(
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
        fold_records.append(
            {
                "fold": fold_idx,
                "pinball": fold_metrics["pinball"],
                "mae": fold_metrics["mae"],
                "mean_u_hat": fold_metrics["mean_u_hat"],
                "coverage": fold_metrics["coverage"],
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
        )

    if np.isnan(oof_pred).any():
        raise VeldraValidationError(
            "OOF predictions contain missing values. Check split configuration."
        )

    mean_metrics = _frontier_metrics(y.to_numpy(), oof_pred, alpha)
    cv_results = pd.DataFrame.from_records(fold_records)

    x_final_train, x_final_valid, y_final_train, y_final_valid = split_for_early_stopping(
        x, y, config
    )
    final_eval_history: dict[str, Any] = {}
    final_model = _train_single_booster(
        x_train=x_final_train,
        y_train=y_final_train,
        x_valid=x_final_valid,
        y_valid=y_final_valid,
        config=config,
        evaluation_history=final_eval_history,
    )
    final_best_iteration, final_num_iterations = _booster_iteration_stats(
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
    }

    metrics = {"folds": fold_records, "mean": mean_metrics}
    feature_schema = {
        "feature_names": x.columns.tolist(),
        "target": config.data.target,
        "task_type": config.task.type,
        "frontier_alpha": alpha,
    }
    return FrontierTrainingOutput(
        model_text=final_model.model_to_string(),
        metrics=metrics,
        cv_results=cv_results,
        feature_schema=feature_schema,
        training_history=training_history,
    )
