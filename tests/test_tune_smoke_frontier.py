from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune


def _frontier_frame(rows: int = 48, seed: int = 4401) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    base = 1.8 + 1.4 * x1 - 0.6 * x2
    y = base + rng.normal(scale=0.25, size=rows) + rng.exponential(scale=0.25, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_smoke_frontier(tmp_path) -> None:
    frame = _frontier_frame()
    path = tmp_path / "train_frontier.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 3},
            "frontier": {"alpha": 0.90},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert result.task_type == "frontier"
    assert result.best_score is not None
    assert result.best_params
    assert result.metadata["metric_name"] == "pinball"
    assert result.metadata["direction"] == "minimize"
