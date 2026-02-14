from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune


def _regression_frame(rows: int = 40, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 2.5 * x1 - 1.2 * x2 + rng.normal(scale=0.3, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_smoke_regression(tmp_path) -> None:
    frame = _regression_frame()
    path = tmp_path / "train_reg.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 3},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert result.task_type == "regression"
    assert result.best_score is not None
    assert result.best_params
    assert result.metadata["metric_name"] == "rmse"
    assert result.metadata["direction"] == "minimize"
