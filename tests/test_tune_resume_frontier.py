from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import tune


def _frontier_frame(rows: int = 40, seed: int = 4501) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.5, 1.5, size=rows)
    x2 = rng.normal(size=rows)
    y = 1.2 + 1.1 * x1 - 0.4 * x2 + rng.normal(scale=0.2, size=rows)
    y = y + rng.exponential(scale=0.15, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_resume_frontier_continues_existing_study(tmp_path) -> None:
    data_path = tmp_path / "frontier.csv"
    _frontier_frame().to_csv(data_path, index=False)
    artifact_dir = tmp_path / "artifacts"

    first = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 9},
            "frontier": {"alpha": 0.90},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "study_name": "frontier_resume",
                "resume": False,
            },
            "export": {"artifact_dir": str(artifact_dir)},
        }
    )
    first_trials = pd.read_parquet(first.metadata["trials_path"])
    assert len(first_trials) == 1
    storage_url = str(first.metadata["storage_url"])
    assert storage_url.startswith("sqlite:///")
    db_path = Path(storage_url.replace("sqlite:///", ""))
    assert db_path.exists()

    second = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 9},
            "frontier": {"alpha": 0.90},
            "tuning": {
                "enabled": True,
                "n_trials": 2,
                "study_name": "frontier_resume",
                "resume": True,
            },
            "export": {"artifact_dir": str(artifact_dir)},
        }
    )
    second_trials = pd.read_parquet(second.metadata["trials_path"])
    assert len(second_trials) >= 3
    assert len(second_trials) > len(first_trials)
