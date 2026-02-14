"""Multiclass classification training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.split import iter_cv_splits


@dataclass(slots=True)
class MulticlassTrainingOutput:
    model_text: str
    metrics: dict[str, Any]
    cv_results: pd.DataFrame
    feature_schema: dict[str, Any]


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
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": config.train.seed,
        **config.train.lgb_params,
    }

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

    return lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[valid_set],
        num_boost_round=300,
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

    for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise VeldraValidationError("Encountered an empty train/valid split.")

        booster = _train_single_booster(
            x_train=x.iloc[train_idx],
            y_train=y.iloc[train_idx],
            x_valid=x.iloc[valid_idx],
            y_valid=y.iloc[valid_idx],
            config=config,
            num_class=num_class,
        )
        pred_raw = np.asarray(
            booster.predict(x.iloc[valid_idx], num_iteration=booster.best_iteration),
            dtype=float,
        )
        pred_proba = _normalize_proba(pred_raw, len(valid_idx), num_class)
        oof_proba[valid_idx, :] = pred_proba

        fold_metrics = _multiclass_metrics(y.iloc[valid_idx].to_numpy(), pred_proba)
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

    final_model = _train_single_booster(
        x_train=x,
        y_train=y,
        x_valid=x,
        y_valid=y,
        config=config,
        num_class=num_class,
    )

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
    )
