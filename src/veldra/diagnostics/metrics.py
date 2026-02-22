"""Task-level diagnostics metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: Any, y_pred: Any, label: str = "overall") -> dict[str, float | str]:
    y_true_f = np.asarray(y_true, dtype=float)
    y_pred_f = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true_f - y_pred_f)
    huber_delta = 1.0
    huber = np.where(
        err < huber_delta,
        0.5 * (err**2),
        huber_delta * (err - 0.5 * huber_delta),
    )
    return {
        "label": label,
        "rmse": float(np.sqrt(mean_squared_error(y_true_f, y_pred_f))),
        "mae": float(mean_absolute_error(y_true_f, y_pred_f)),
        "mape": float(mean_absolute_percentage_error(y_true_f, y_pred_f)),
        "r2": float(r2_score(y_true_f, y_pred_f)),
        "huber": float(np.mean(huber)),
    }


def binary_metrics(y_true: Any, y_score: Any, label: str = "overall") -> dict[str, float | str]:
    y_true_i = np.asarray(y_true, dtype=int)
    score = np.clip(np.asarray(y_score, dtype=float), 1e-7, 1.0 - 1e-7)
    label_pred = (score >= 0.5).astype(int)
    return {
        "label": label,
        "auc": float(roc_auc_score(y_true_i, score)),
        "logloss": float(log_loss(y_true_i, score, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true_i, score)),
        "average_precision": float(average_precision_score(y_true_i, score)),
        "accuracy": float(accuracy_score(y_true_i, label_pred)),
        "f1": float(f1_score(y_true_i, label_pred, zero_division=0)),
    }


def multiclass_metrics(
    y_true: Any,
    y_proba: Any,
    label: str = "overall",
) -> dict[str, float | str]:
    y_true_i = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    proba = np.clip(proba, 1e-7, 1.0 - 1e-7)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_pred = np.argmax(proba, axis=1)
    n_classes = proba.shape[1]
    return {
        "label": label,
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "macro_f1": float(f1_score(y_true_i, y_pred, average="macro")),
        "multi_logloss": float(log_loss(y_true_i, proba, labels=list(range(n_classes)))),
        "multi_error": float(1.0 - accuracy_score(y_true_i, y_pred)),
    }


def frontier_metrics(
    y_true: Any,
    y_pred: Any,
    alpha: float,
    label: str = "overall",
) -> dict[str, float | str]:
    y_true_f = np.asarray(y_true, dtype=float)
    y_pred_f = np.asarray(y_pred, dtype=float)
    err = y_true_f - y_pred_f
    pinball = np.maximum(alpha * err, (alpha - 1.0) * err)
    return {
        "label": label,
        "pinball": float(np.mean(pinball)),
        "mae": float(mean_absolute_error(y_true_f, y_pred_f)),
        "coverage": float(np.mean(y_true_f <= y_pred_f)),
    }


def split_in_out_metrics(
    metric_fn: Callable[..., dict[str, Any]],
    y_true: Any,
    y_pred: Any,
    fold_ids: Any,
    eval_fold_ids: set[int] | list[int],
) -> pd.DataFrame:
    fold = np.asarray(fold_ids)
    eval_set = set(int(v) for v in eval_fold_ids)
    out_mask = np.array([int(v) in eval_set for v in fold], dtype=bool)
    in_mask = ~out_mask

    records = [
        metric_fn(np.asarray(y_true)[in_mask], np.asarray(y_pred)[in_mask], label="in_sample"),
        metric_fn(
            np.asarray(y_true)[out_mask],
            np.asarray(y_pred)[out_mask],
            label="out_of_sample",
        ),
    ]
    return pd.DataFrame.from_records(records)
