from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune


def _binary_frame(rows: int = 48, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 1.4 * x1 - 1.0 * x2 + rng.normal(scale=0.35, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_smoke_binary(tmp_path) -> None:
    frame = _binary_frame()
    path = tmp_path / "train_bin.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 11},
            "postprocess": {"threshold_optimization": {"enabled": True, "objective": "f1"}},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert result.task_type == "binary"
    assert result.best_score is not None
    assert result.best_params
    assert result.metadata["metric_name"] == "auc"
    assert result.metadata["direction"] == "maximize"
