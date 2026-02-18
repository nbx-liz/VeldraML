from __future__ import annotations

import math

import numpy as np

from veldra.api import fit
from veldra.modeling.binary import _binary_label_metrics


def test_binary_label_metrics_with_constant_scores_and_threshold_bounds() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    p_pred = np.full(4, 0.5, dtype=float)

    metrics_low = _binary_label_metrics(y_true, p_pred, threshold=0.0)
    metrics_high = _binary_label_metrics(y_true, p_pred, threshold=1.0)

    for payload in (metrics_low, metrics_high):
        for value in payload.values():
            assert math.isfinite(float(value))
            assert 0.0 <= float(value) <= 1.0


def test_binary_fit_handles_nan_feature_values(tmp_path, binary_frame) -> None:
    frame = binary_frame(rows=120, seed=601, coef1=1.4, coef2=-1.1, noise=0.4)
    frame.loc[0, "x1"] = np.nan
    frame.loc[11, "x2"] = np.nan
    path = tmp_path / "binary_nan_feature.csv"
    frame.to_csv(path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 21},
            "postprocess": {"calibration": "platt", "threshold": 0.5},
            "train": {"num_boost_round": 40, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert {"auc", "logloss", "brier"} <= set(run.metrics)


def test_binary_fit_with_extreme_imbalance_keeps_metrics_defined(tmp_path) -> None:
    rng = np.random.default_rng(602)
    rows = 1200
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 1.7 * x1 - 0.9 * x2 + rng.normal(scale=0.3, size=rows)
    threshold = np.quantile(score, 0.98)
    target = (score >= threshold).astype(int)

    frame = np.column_stack([x1, x2, target])
    path = tmp_path / "binary_imbalance.csv"
    np.savetxt(path, frame, delimiter=",", header="x1,x2,target", comments="")

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 33},
            "postprocess": {"calibration": "platt"},
            "train": {"num_boost_round": 40, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    for key in ("auc", "logloss", "brier"):
        assert math.isfinite(float(run.metrics[key])), key
