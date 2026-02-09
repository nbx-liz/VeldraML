from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune


def _regression_frame(rows: int = 36, seed: int = 202) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 2.2 * x1 - 0.9 * x2 + rng.normal(scale=0.3, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _binary_frame(rows: int = 42, seed: int = 203) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    z = 1.1 * x1 - 1.0 * x2 + rng.normal(scale=0.4, size=rows)
    y = (z > np.median(z)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _multiclass_frame(rows_per_class: int = 12, seed: int = 204) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["a", "b", "c"]
    chunks: list[pd.DataFrame] = []
    for i, label in enumerate(labels):
        base = i * 1.6
        x1 = rng.normal(loc=base, scale=0.4, size=rows_per_class)
        x2 = rng.normal(loc=1.2 - base, scale=0.4, size=rows_per_class)
        chunks.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(chunks, ignore_index=True)


def test_tune_supports_task_constrained_objective_selection(tmp_path) -> None:
    reg_path = tmp_path / "reg.csv"
    bin_path = tmp_path / "bin.csv"
    mc_path = tmp_path / "mc.csv"
    _regression_frame().to_csv(reg_path, index=False)
    _binary_frame().to_csv(bin_path, index=False)
    _multiclass_frame().to_csv(mc_path, index=False)

    reg = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(reg_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "mae"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert reg.metadata["metric_name"] == "mae"

    binary = tune(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(bin_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 1},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "logloss"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert binary.metadata["metric_name"] == "logloss"

    multiclass = tune(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(mc_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 1},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "accuracy"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert multiclass.metadata["metric_name"] == "accuracy"

