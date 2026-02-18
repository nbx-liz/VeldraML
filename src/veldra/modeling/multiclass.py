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
from veldra.modeling._cv_runner import TaskSpec, booster_iteration_stats, run_cv_training
from veldra.modeling.utils import (
    resolve_auto_num_leaves,
    resolve_feature_weights,
    resolve_ratio_params,
)
from veldra.split import iter_cv_splits


@dataclass(slots=True)
class MulticlassTrainingOutput:
    model_text: str
    metrics: dict[str, Any]
    cv_results: pd.DataFrame
    feature_schema: dict[str, Any]
    training_history: dict[str, Any]
    observation_table: pd.DataFrame


def _booster_iteration_stats(booster: Any, fallback_rounds: int) -> tuple[int, int]:
    return booster_iteration_stats(booster, fallback_rounds)


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
    metric_value: str | list[str] = config.train.metrics or "multi_logloss"
    params = {
        "objective": "multiclass",
        "num_class": num_class,
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
    accuracy = float(accuracy_score(y_true, y_pred))
    logloss_value = float(log_loss(y_true, proba, labels=labels))
    error_rate = float(1.0 - accuracy)
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": logloss_value,
        "multi_logloss": logloss_value,
        "error_rate": error_rate,
        "multi_error": error_rate,
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

    def _fit_booster(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_valid: pd.DataFrame,
        y_valid: pd.Series,
        cfg: RunConfig,
        ctx: dict[str, Any],
        evaluation_history: dict[str, Any],
    ) -> lgb.Booster:
        return _train_single_booster(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            config=cfg,
            num_class=int(ctx["num_class"]),
            evaluation_history=evaluation_history,
        )

    def _predict_valid(
        booster: lgb.Booster,
        x_valid: pd.DataFrame,
        ctx: dict[str, Any],
    ) -> np.ndarray:
        pred_raw = np.asarray(
            booster.predict(x_valid, num_iteration=booster.best_iteration),
            dtype=float,
        )
        return _normalize_proba(pred_raw, len(x_valid), int(ctx["num_class"]))

    def _fold_metrics(
        y_valid: pd.Series,
        pred: np.ndarray,
        _ctx: dict[str, Any],
    ) -> dict[str, float]:
        return _multiclass_metrics(y_valid.to_numpy(), pred)

    def _fold_record(
        fold_idx: int,
        metrics: dict[str, float],
        n_train: int,
        n_valid: int,
        _ctx: dict[str, Any],
    ) -> dict[str, float | int]:
        return {
            "fold": fold_idx,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "logloss": metrics["logloss"],
            "multi_error": metrics["multi_error"],
            "n_train": n_train,
            "n_valid": n_valid,
        }

    def _mean_metrics(y_all: pd.Series, oof: np.ndarray, _ctx: dict[str, Any]) -> dict[str, float]:
        return _multiclass_metrics(y_all.to_numpy(), oof)

    def _observation_table(
        fold_ids: np.ndarray,
        y_all: pd.Series,
        oof: np.ndarray,
        ctx: dict[str, Any],
    ) -> pd.DataFrame:
        target_classes_local = list(ctx["target_classes"])
        oof_pred_idx = np.argmax(oof, axis=1)
        table = pd.DataFrame(
            {
                "fold_id": fold_ids,
                "in_out_label": np.where(fold_ids > 0, "out_of_fold", "in_fold"),
                "y_true": y_all.to_numpy(dtype=int),
                "label_pred": oof_pred_idx.astype(int),
            }
        )
        for idx, class_label in enumerate(target_classes_local):
            table[f"proba_{class_label}"] = oof[:, idx]
        return table

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
            init_oof=lambda n_rows, ctx: np.full(
                (n_rows, int(ctx["num_class"])),
                np.nan,
                dtype=float,
            ),
        ),
        context={"num_class": num_class, "target_classes": target_classes},
    )

    metrics = {
        "folds": cv_out.fold_records,
        "mean": cv_out.mean_metrics,
    }
    feature_schema = {
        "feature_names": x.columns.tolist(),
        "target": config.data.target,
        "task_type": config.task.type,
        "target_classes": target_classes,
    }
    return MulticlassTrainingOutput(
        model_text=cv_out.final_model.model_to_string(),
        metrics=metrics,
        cv_results=cv_out.cv_results,
        feature_schema=feature_schema,
        training_history=cv_out.training_history,
        observation_table=cv_out.observation_table,
    )
