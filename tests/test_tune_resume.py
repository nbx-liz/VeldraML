from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veldra.api import tune
from veldra.api.exceptions import VeldraValidationError


def _regression_frame(rows: int = 32, seed: int = 405) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.7 * x1 - 1.0 * x2 + rng.normal(scale=0.25, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_resume_continues_same_study(tmp_path) -> None:
    data_path = tmp_path / "train.csv"
    _regression_frame().to_csv(data_path, index=False)
    artifact_dir = tmp_path / "artifacts"

    first = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 9},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "study_name": "resume_demo",
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
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 9},
            "tuning": {
                "enabled": True,
                "n_trials": 2,
                "study_name": "resume_demo",
                "resume": True,
            },
            "export": {"artifact_dir": str(artifact_dir)},
        }
    )
    second_trials = pd.read_parquet(second.metadata["trials_path"])
    assert len(second_trials) >= 3
    assert len(second_trials) > len(first_trials)


def test_tune_existing_study_requires_resume_true(tmp_path) -> None:
    data_path = tmp_path / "train.csv"
    _regression_frame(seed=406).to_csv(data_path, index=False)
    artifact_dir = tmp_path / "artifacts"

    payload = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": str(data_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 9},
        "tuning": {
            "enabled": True,
            "n_trials": 1,
            "study_name": "duplicate_demo",
            "resume": False,
        },
        "export": {"artifact_dir": str(artifact_dir)},
    }
    tune(payload)
    with pytest.raises(VeldraValidationError, match="already exists"):
        tune(payload)
