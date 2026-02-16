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


def _frontier_frame(rows: int = 30, seed: int = 205) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    y = 1.3 + 1.1 * x1 - 0.5 * x2 + rng.normal(scale=0.2, size=rows)
    y = y + rng.exponential(scale=0.2, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_supports_task_constrained_objective_selection(tmp_path) -> None:
    reg_path = tmp_path / "reg.csv"
    bin_path = tmp_path / "bin.csv"
    mc_path = tmp_path / "mc.csv"
    fr_path = tmp_path / "fr.csv"
    _regression_frame().to_csv(reg_path, index=False)
    _binary_frame().to_csv(bin_path, index=False)
    _multiclass_frame().to_csv(mc_path, index=False)
    _frontier_frame().to_csv(fr_path, index=False)
    causal_frame = _regression_frame(seed=909)
    causal_frame["treatment"] = (causal_frame["x1"] > float(causal_frame["x1"].median())).astype(
        int
    )
    causal_path = tmp_path / "causal.csv"
    causal_frame.to_csv(causal_path, index=False)

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

    binary_topk = tune(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(bin_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 1},
            "train": {"top_k": 5},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "precision_at_k"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert binary_topk.metadata["metric_name"] == "precision_at_k"

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

    frontier = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(fr_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "frontier": {"alpha": 0.90},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "pinball"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert frontier.metadata["metric_name"] == "pinball"

    frontier_penalty = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(fr_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "frontier": {"alpha": 0.90},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "objective": "pinball_coverage_penalty",
                "coverage_target": 0.91,
                "coverage_tolerance": 0.02,
                "penalty_weight": 1.5,
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert frontier_penalty.metadata["metric_name"] == "pinball_coverage_penalty"

    causal = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(causal_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "causal": {"method": "dr", "treatment_col": "treatment"},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "dr_std_error"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert causal.metadata["metric_name"] == "dr_std_error"

    causal_balance = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(causal_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "causal": {"method": "dr", "treatment_col": "treatment"},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "objective": "dr_balance_priority",
                "causal_balance_threshold": 0.10,
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert causal_balance.metadata["metric_name"] == "dr_balance_priority"
