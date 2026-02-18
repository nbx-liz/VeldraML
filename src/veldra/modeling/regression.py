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
from veldra.modeling._cv_runner import TaskSpec, booster_iteration_stats, run_cv_training
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
    return booster_iteration_stats(booster, fallback_rounds)


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

    def _fit_booster(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        cfg: RunConfig,
        _ctx: dict[str, Any],
        evaluation_history: dict[str, Any],
    ) -> lgb.Booster:
        return _train_single_booster(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            config=cfg,
            evaluation_history=evaluation_history,
        )

    def _predict_valid(
        booster: lgb.Booster,
        x_valid: pd.DataFrame,
        _ctx: dict[str, Any],
    ) -> np.ndarray:
        return np.asarray(
            booster.predict(x_valid, num_iteration=booster.best_iteration),
            dtype=float,
        )

    def _fold_metrics(
        y_valid: pd.Series,
        pred: np.ndarray,
        _ctx: dict[str, Any],
    ) -> dict[str, float]:
        y_true = y_valid.to_numpy(dtype=float)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred)),
            "mape": float(mean_absolute_percentage_error(y_true, pred)),
            "r2": float(r2_score(y_true, pred)),
        }

    def _fold_record(
        fold_idx: int,
        metrics: dict[str, float],
        n_train: int,
        n_valid: int,
        _ctx: dict[str, Any],
    ) -> dict[str, float | int]:
        return {
            "fold": fold_idx,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "r2": metrics["r2"],
            "n_train": n_train,
            "n_valid": n_valid,
        }

    def _mean_metrics(y_all: pd.Series, oof: np.ndarray, _ctx: dict[str, Any]) -> dict[str, float]:
        y_true = y_all.to_numpy(dtype=float)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, oof))),
            "mae": float(mean_absolute_error(y_true, oof)),
            "mape": float(mean_absolute_percentage_error(y_true, oof)),
            "r2": float(r2_score(y_true, oof)),
        }

    def _observation_table(
        fold_ids: np.ndarray,
        y_all: pd.Series,
        oof: np.ndarray,
        _ctx: dict[str, Any],
    ) -> pd.DataFrame:
        y_true = y_all.to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "fold_id": fold_ids,
                "in_out_label": np.where(fold_ids > 0, "out_of_fold", "in_fold"),
                "y_true": y_true,
                "prediction": oof,
                "residual": y_true - oof,
            }
        )

    cv_out = run_cv_training(
        config=config,
        x=x,
        y=y,
        splits=splits,
        spec=TaskSpec(
            fit_booster=_fit_booster,
            predict_valid=_predict_valid,
            fold_metrics=_fold_metrics,
            fold_record=_fold_record,
            mean_metrics=_mean_metrics,
            observation_table=_observation_table,
            init_oof=lambda n_rows, _ctx: np.full(n_rows, np.nan, dtype=float),
        ),
        split_for_early_stopping_fn=split_for_early_stopping,
    )

    metrics = {
        "folds": cv_out.fold_records,
        "mean": cv_out.mean_metrics,
    }
    feature_schema = {
        "feature_names": x.columns.tolist(),
        "target": config.data.target,
        "task_type": config.task.type,
    }
    return RegressionTrainingOutput(
        model_text=cv_out.final_model.model_to_string(),
        metrics=metrics,
        cv_results=cv_out.cv_results,
        feature_schema=feature_schema,
        training_history=cv_out.training_history,
        observation_table=cv_out.observation_table,
    )
