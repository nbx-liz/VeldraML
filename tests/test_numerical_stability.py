from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veldra.api import estimate_dr
from veldra.diagnostics.metrics import binary_metrics, frontier_metrics, regression_metrics


def test_diagnostics_metrics_are_finite_for_extreme_probabilities() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    y_score = np.array([0.001, 0.999, 0.002, 0.998, 0.003, 0.997], dtype=float)
    metrics = binary_metrics(y_true, y_score, label="extreme")
    for key, value in metrics.items():
        if key == "label":
            continue
        assert math.isfinite(float(value)), key


def test_frontier_metrics_are_finite_on_alpha_boundaries() -> None:
    y_true = np.array([1.0, 1.3, 1.6, 1.9, 2.2], dtype=float)
    y_pred = np.array([1.1, 1.4, 1.5, 2.0, 2.1], dtype=float)
    for alpha in (0.01, 0.99):
        metrics = frontier_metrics(y_true, y_pred, alpha=alpha, label=f"a={alpha}")
        assert math.isfinite(float(metrics["pinball"]))
        assert math.isfinite(float(metrics["mae"]))
        assert 0.0 <= float(metrics["coverage"]) <= 1.0


def test_regression_metrics_are_finite_with_small_target_scale() -> None:
    y_true = np.array([1e-6, 2e-6, 3e-6, 4e-6], dtype=float)
    y_pred = np.array([1.1e-6, 1.9e-6, 2.9e-6, 4.1e-6], dtype=float)
    metrics = regression_metrics(y_true, y_pred, label="tiny")
    assert math.isfinite(float(metrics["rmse"]))
    assert math.isfinite(float(metrics["mae"]))
    assert math.isfinite(float(metrics["mape"]))


def test_causal_dr_stability_with_extreme_propensity_scores(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 220
    x = rng.normal(size=n)
    # Strong separation to induce extreme propensity.
    treatment = (x > 0.9).astype(int)
    treatment[:8] = 1
    treatment[-8:] = 0
    outcome = 2.0 + 0.8 * x + 1.5 * treatment + rng.normal(scale=0.2, size=n)
    frame = pd.DataFrame({"x": x, "treatment": treatment, "target": outcome})
    path = tmp_path / "dr_extreme.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 42},
            "causal": {
                "method": "dr",
                "treatment_col": "treatment",
                "estimand": "att",
                "propensity_clip": 0.001,
                "cross_fit": True,
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert math.isfinite(float(result.estimate))
    assert math.isfinite(float(result.metrics["overlap_metric"]))
    assert math.isfinite(float(result.metrics["smd_max_weighted"]))


def test_metrics_nan_propagation_contract() -> None:
    with pytest.raises(ValueError):
        regression_metrics(
            np.array([1.0, 2.0, 3.0], dtype=float),
            np.array([1.1, np.nan, 2.9], dtype=float),
            label="nan-case",
        )

    with pytest.raises(ValueError):
        binary_metrics(
            np.array([0, 1, 0, 1], dtype=int),
            np.array([0.1, np.nan, 0.2, 0.9], dtype=float),
            label="nan-case",
        )

    with pytest.raises(ValueError):
        frontier_metrics(
            np.array([1.0, 2.0, 3.0], dtype=float),
            np.array([1.1, np.nan, 2.9], dtype=float),
            alpha=0.9,
            label="nan-case",
        )


def test_binary_metrics_remain_finite_under_extreme_clipping_edges() -> None:
    metrics = binary_metrics(
        np.array([0, 1, 0, 1, 0, 1], dtype=int),
        np.array([0.0, 1.0, 1e-15, 1.0 - 1e-15, 1e-12, 1.0], dtype=float),
        label="clip-boundary",
    )
    for key, value in metrics.items():
        if key == "label":
            continue
        assert math.isfinite(float(value)), key
