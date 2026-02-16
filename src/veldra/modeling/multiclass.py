"""Multiclass classification training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight

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
class MulticlassTrainingOutput:
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


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_feature_frame(
    config: RunConfig,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[Any]]:
    target = config.data.target
    if target not in data.columns:
        raise VeldraValidationError(f"Target column '{target}' was not found in input data.")
    if data.empty:
        raise VeldraValidationError("Input data is empty.")

    y = data[target].copy()
    if y.isna().any():
        raise VeldraValidationError("Target column contains null values.")

    unique = pd.unique(y)
    if len(unique) < 3:
        raise VeldraValidationError("Multiclass task requires at least three target classes.")
    target_classes = sorted((_to_python_scalar(v) for v in unique), key=lambda v: str(v))
    class_mapping = {label: idx for idx, label in enumerate(target_classes)}
    y_encoded = y.map(class_mapping)
    if y_encoded.isna().any():
        raise VeldraValidationError("Failed to encode multiclass target labels.")

    drop_cols = set(config.data.drop_cols + config.data.id_cols + [target])
    feature_cols = [col for col in data.columns if col not in drop_cols]
    if not feature_cols:
        raise VeldraValidationError(
            "No training features remain after applying drop/id/target columns."
        )

    x = data.loc[:, feature_cols].copy()
    return x, y_encoded.astype(int), target_classes


def _train_single_booster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: RunConfig,
    num_class: int,
    evaluation_history: dict[str, Any] | None = None,
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
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
    train_weight: np.ndarray | None = None
    if config.train.class_weight is not None:
        train_weight = np.asarray(
            [float(config.train.class_weight.get(str(int(label)), 1.0)) for label in y_train],
            dtype=float,
        )
    elif config.train.auto_class_weight:
        train_weight = np.asarray(compute_sample_weight("balanced", y_train), dtype=float)

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weight,
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


def _normalize_proba(raw: np.ndarray, n_rows: int, num_class: int) -> np.ndarray:
    if raw.ndim == 1:
        if raw.size != n_rows * num_class:
            raise VeldraValidationError(
                "Multiclass prediction output has invalid shape for configured classes."
            )
        raw = raw.reshape(n_rows, num_class)
    if raw.ndim != 2 or raw.shape[1] != num_class:
        raise VeldraValidationError("Multiclass prediction output has invalid dimensions.")

    proba = np.clip(raw.astype(float), 1e-7, 1 - 1e-7)
    row_sum = proba.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise VeldraValidationError("Multiclass prediction probabilities have invalid row sums.")
    return proba / row_sum


def _multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_pred = np.argmax(proba, axis=1)
    labels = list(range(proba.shape[1]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": float(log_loss(y_true, proba, labels=labels)),
    }


def train_multiclass_with_cv(config: RunConfig, data: pd.DataFrame) -> MulticlassTrainingOutput:
    """Train multiclass model with CV and return artifact payload.

    Notes
    -----
    - Class labels are mapped to contiguous indices and restored through
      ``feature_schema.target_classes``.
    - Fold-level probabilities are normalized and merged into OOF probabilities
      before metric aggregation.
    - Shape and probability-sum checks guard against malformed model outputs.
    """
    if config.task.type != "multiclass":
        raise VeldraValidationError(
            "train_multiclass_with_cv only supports task.type='multiclass'."
        )
    if not config.data.path:
        raise VeldraValidationError("data.path is required for fit.")

    if config.split.type == "timeseries":
        data = data.sort_values(config.split.time_col).reset_index(drop=True)

    x, y, target_classes = _build_feature_frame(config, data)
    num_class = len(target_classes)
    splits = iter_cv_splits(config, data, x, y)

    oof_proba = np.full((len(x), num_class), np.nan, dtype=float)
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
            num_class=num_class,
            evaluation_history=eval_history,
        )
        pred_raw = np.asarray(
            booster.predict(x.iloc[valid_idx], num_iteration=booster.best_iteration),
            dtype=float,
        )
        pred_proba = _normalize_proba(pred_raw, len(valid_idx), num_class)
        oof_proba[valid_idx, :] = pred_proba

        fold_metrics = _multiclass_metrics(y.iloc[valid_idx].to_numpy(), pred_proba)
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
                "accuracy": fold_metrics["accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
                "logloss": fold_metrics["logloss"],
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
        )

    if np.isnan(oof_proba).any():
        raise VeldraValidationError(
            "OOF predictions contain missing values. Check split configuration."
        )

    mean_metrics = _multiclass_metrics(y.to_numpy(), oof_proba)
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
        num_class=num_class,
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
        "mean": mean_metrics,
    }
    feature_schema = {
        "feature_names": x.columns.tolist(),
        "target": config.data.target,
        "task_type": config.task.type,
        "target_classes": target_classes,
    }
    return MulticlassTrainingOutput(
        model_text=final_model.model_to_string(),
        metrics=metrics,
        cv_results=cv_results,
        feature_schema=feature_schema,
        training_history=training_history,
    )
