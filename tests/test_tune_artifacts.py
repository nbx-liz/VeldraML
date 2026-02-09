from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import tune


def _regression_frame(rows: int = 36, seed: int = 113) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.8 * x1 - 0.7 * x2 + rng.normal(scale=0.25, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_writes_study_summary_and_trials(tmp_path) -> None:
    frame = _regression_frame()
    path = tmp_path / "reg.csv"
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

    summary_path = Path(result.metadata["summary_path"])
    trials_path = Path(result.metadata["trials_path"])
    assert summary_path.exists()
    assert trials_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == result.run_id
    assert summary["task_type"] == "regression"
    assert summary["metric_name"] == "rmse"
    assert summary["direction"] == "minimize"
    assert summary["best_params"]

    trials = pd.read_parquet(trials_path)
    assert len(trials) >= 1
    assert {"number", "value", "state"} <= set(trials.columns)

