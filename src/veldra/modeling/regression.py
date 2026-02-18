"""Regression training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

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
class RegressionTrainingOutput:
    model_text: str
    metrics: dict[str, Any]
    cv_results: pd.DataFrame
    feature_schema: dict[str, Any]
    training_history: dict[str, Any]
    observation_table: pd.DataFrame


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

    drop_cols = set(config.data.drop_cols + config.data.id_cols + [target])
    feature_cols = [col for col in data.columns if col not in drop_cols]
    if not feature_cols:
        raise VeldraValidationError(
            "No training features remain after applying drop/id/target columns."
        )

    x = data.loc[:, feature_cols].copy()
    y = data[target].copy()
    return x, y


def _train_single_booster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: RunConfig,
    evaluation_history: dict[str, Any] | None = None,
) -> lgb.Booster:
    metric_value: str | list[str] = config.train.metrics or "rmse"
    params = {
        "objective": "regression",
        "metric": metric_value,
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

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=[col for col in config.data.categorical if col in x_train.columns],
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        categorical_feature=[col for col in config.data.categorical if col in x_valid.columns],
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


def train_regression_with_cv(config: RunConfig, data: pd.DataFrame) -> RegressionTrainingOutput:
    """Train regression model with CV and return artifact payload.

    Notes
    -----
    - Data is split by ``split.type`` and fold-level out-of-fold predictions are
      aggregated into mean metrics.
    - For ``timeseries`` split, data is ordered by ``split.time_col`` before
      fold construction to avoid temporal leakage.
    - Training fails fast on empty folds or missing OOF predictions.
    """
    if config.task.type != "regression":
        raise VeldraValidationError(
            "train_regression_with_cv only supports task.type='regression'."
        )
    if not config.data.path:
        raise VeldraValidationError("data.path is required for fit.")

    if config.split.type == "timeseries":
        data = data.sort_values(config.split.time_col).reset_index(drop=True)

    x, y = _build_feature_frame(config, data)
    if config.split.type == "stratified":
        raise VeldraValidationError("split.type='stratified' is not supported for regression task.")
    splits = iter_cv_splits(config, data, x)

    oof_pred = np.full(len(x), np.nan, dtype=float)
    fold_ids = np.full(len(x), -1, dtype=int)
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
        pred = booster.predict(x.iloc[valid_idx], num_iteration=booster.best_iteration)
        oof_pred[valid_idx] = pred
        fold_ids[valid_idx] = fold_idx
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

        fold_rmse = float(np.sqrt(mean_squared_error(y.iloc[valid_idx], pred)))
        fold_mae = float(mean_absolute_error(y.iloc[valid_idx], pred))
        fold_mape = float(mean_absolute_percentage_error(y.iloc[valid_idx], pred))
        fold_r2 = float(r2_score(y.iloc[valid_idx], pred))
        fold_records.append(
            {
                "fold": fold_idx,
                "rmse": fold_rmse,
                "mae": fold_mae,
                "mape": fold_mape,
                "r2": fold_r2,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
        )

    if np.isnan(oof_pred).any():
        raise VeldraValidationError(
            "OOF predictions contain missing values. Check split configuration."
        )

    mean_rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
    mean_mae = float(mean_absolute_error(y, oof_pred))
    mean_mape = float(mean_absolute_percentage_error(y, oof_pred))
    mean_r2 = float(r2_score(y, oof_pred))
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

    metrics = {
        "folds": fold_records,
        "mean": {"rmse": mean_rmse, "mae": mean_mae, "mape": mean_mape, "r2": mean_r2},
    }
    observation_table = pd.DataFrame(
        {
            "fold_id": fold_ids,
            "in_out_label": np.where(fold_ids > 0, "out_of_fold", "in_fold"),
            "y_true": y.to_numpy(dtype=float),
            "prediction": oof_pred,
            "residual": y.to_numpy(dtype=float) - oof_pred,
        }
    )
    feature_schema = {
        "feature_names": x.columns.tolist(),
        "target": config.data.target,
        "task_type": config.task.type,
    }
    return RegressionTrainingOutput(
        model_text=final_model.model_to_string(),
        metrics=metrics,
        cv_results=cv_results,
        feature_schema=feature_schema,
        training_history=training_history,
        observation_table=observation_table,
    )
