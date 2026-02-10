from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune


def _frontier_frame(rows: int = 42, seed: int = 5101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    y = 1.4 + 1.0 * x1 - 0.6 * x2 + rng.normal(scale=0.2, size=rows)
    y = y + rng.exponential(scale=0.2, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_frontier_tune_default_objective_remains_pinball(tmp_path) -> None:
    frame = _frontier_frame()
    path = tmp_path / "frontier_default.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 11},
            "frontier": {"alpha": 0.90},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert result.metadata["metric_name"] == "pinball"
    assert result.metadata["coverage_target"] == 0.90
    assert result.metadata["coverage_tolerance"] == 0.01
    assert result.metadata["penalty_weight"] == 1.0


def test_frontier_tune_accepts_coverage_penalty_objective(tmp_path) -> None:
    frame = _frontier_frame(seed=5102)
    path = tmp_path / "frontier_penalty.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 12},
            "frontier": {"alpha": 0.90},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "preset": "fast",
                "objective": "pinball_coverage_penalty",
                "coverage_target": 0.93,
                "coverage_tolerance": 0.03,
                "penalty_weight": 2.5,
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert result.metadata["metric_name"] == "pinball_coverage_penalty"
    assert result.metadata["coverage_target"] == 0.93
    assert result.metadata["coverage_tolerance"] == 0.03
    assert result.metadata["penalty_weight"] == 2.5
