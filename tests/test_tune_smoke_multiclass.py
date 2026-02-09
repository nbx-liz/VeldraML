from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune


def _multiclass_frame(rows_per_class: int = 18, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["alpha", "beta", "gamma"]
    frames: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        center = float(idx) * 1.7
        x1 = rng.normal(loc=center, scale=0.5, size=rows_per_class)
        x2 = rng.normal(loc=1.8 - center, scale=0.5, size=rows_per_class)
        frames.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(frames, ignore_index=True)


def test_tune_smoke_multiclass(tmp_path) -> None:
    frame = _multiclass_frame()
    path = tmp_path / "train_mc.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 5},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert result.task_type == "multiclass"
    assert result.best_score is not None
    assert result.best_params
    assert result.metadata["metric_name"] == "macro_f1"
    assert result.metadata["direction"] == "maximize"

