"""LightGBM native SHAP wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class _ShapeError(ValueError):
    pass


def _resolve_booster(booster: Any) -> Any:
    if hasattr(booster, "booster_"):
        return booster.booster_
    return booster


def _to_contrib_matrix(raw: np.ndarray, n_rows: int, n_features: int) -> np.ndarray:
    if raw.ndim != 2:
        raise _ShapeError("Expected 2D pred_contrib output.")
    if raw.shape[1] == n_features + 1:
        return raw[:, :-1]
    if raw.shape[1] % (n_features + 1) != 0:
        raise _ShapeError("Unexpected pred_contrib width.")
    n_classes = raw.shape[1] // (n_features + 1)
    reshaped = raw.reshape(n_rows, n_classes, n_features + 1)
    return reshaped[:, 0, :-1]


def compute_shap(booster: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Compute native SHAP (pred_contrib) as a frame aligned to X columns."""
    resolved = _resolve_booster(booster)
    raw = np.asarray(resolved.predict(X, pred_contrib=True), dtype=float)
    contrib = _to_contrib_matrix(raw, len(X), X.shape[1])
    return pd.DataFrame(contrib, columns=X.columns, index=X.index)


def compute_shap_multiclass(
    booster: Any,
    X: pd.DataFrame,
    predictions: np.ndarray,
    n_classes: int,
) -> pd.DataFrame:
    """Compute class-conditional SHAP selecting each row's predicted class."""
    resolved = _resolve_booster(booster)
    raw = np.asarray(resolved.predict(X, pred_contrib=True), dtype=float)
    n_rows = len(X)
    n_features = X.shape[1]

    if raw.ndim != 2 or raw.shape[1] != (n_features + 1) * int(n_classes):
        raise ValueError("Unexpected multiclass pred_contrib shape.")

    reshaped = raw.reshape(n_rows, int(n_classes), n_features + 1)
    pred_idx = np.asarray(predictions, dtype=int)
    if pred_idx.shape[0] != n_rows:
        raise ValueError("predictions length must match X rows.")

    out = np.zeros((n_rows, n_features), dtype=float)
    for row in range(n_rows):
        klass = int(pred_idx[row])
        out[row, :] = reshaped[row, klass, :-1]

    return pd.DataFrame(out, columns=X.columns, index=X.index)
