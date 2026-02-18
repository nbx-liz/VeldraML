from __future__ import annotations

import math

import numpy as np
import pandas as pd

from veldra.api import fit


def test_regression_fit_with_tiny_target_scale(tmp_path) -> None:
    rng = np.random.default_rng(701)
    rows = 80
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    target = (2.0 * x1 - 0.8 * x2 + rng.normal(scale=0.1, size=rows)) * 1e-6
    frame = pd.DataFrame({"x1": x1, "x2": x2, "target": target})
    path = tmp_path / "reg_tiny_scale.csv"
    frame.to_csv(path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 41},
            "train": {"num_boost_round": 40, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    for key in ("rmse", "mae", "mape", "r2"):
        assert math.isfinite(float(run.metrics[key])), key


def test_regression_fit_with_constant_target(tmp_path, regression_frame) -> None:
    frame = regression_frame(rows=72, seed=702, coef1=0.0, coef2=0.0, noise=0.0)
    frame["target"] = 7.0
    path = tmp_path / "reg_constant_target.csv"
    frame.to_csv(path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 42},
            "train": {"num_boost_round": 35, "early_stopping_rounds": 6},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert run.task_type == "regression"
    assert math.isfinite(float(run.metrics["rmse"]))


def test_regression_fit_with_outlier_mix(tmp_path, outlier_frame) -> None:
    frame = outlier_frame.copy()
    path = tmp_path / "reg_outlier.csv"
    frame.to_csv(path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 43},
            "train": {"num_boost_round": 45, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert run.metrics["rmse"] >= 0.0
    assert run.metrics["mae"] >= 0.0
