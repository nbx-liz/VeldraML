from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from veldra.api import tune
from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import tuning


def _regression_frame(rows: int = 36, seed: int = 901) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.6 * x1 - 1.1 * x2 + rng.normal(scale=0.25, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_duplicate_study_requires_resume_true(tmp_path: Path) -> None:
    frame = _regression_frame(seed=902)
    data_path = tmp_path / "tune_edge_train.csv"
    frame.to_csv(data_path, index=False)

    payload = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": str(data_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 31},
        "tuning": {
            "enabled": True,
            "n_trials": 1,
            "study_name": "edge_duplicate_study",
            "resume": False,
        },
        "export": {"artifact_dir": str(tmp_path / "artifacts")},
    }
    tune(payload)
    with pytest.raises(VeldraValidationError, match="already exists"):
        tune(payload)


def test_build_trial_config_rejects_unknown_train_field() -> None:
    config = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": "artifacts"},
        }
    )
    with pytest.raises(VeldraValidationError, match="unknown TrainConfig field"):
        tuning._build_trial_config(config, {"train.not_a_field": 1})


def test_run_tuning_raises_when_no_trial_completes(monkeypatch, tmp_path: Path) -> None:
    config = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    frame = _regression_frame(seed=903)

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[Any] = []

        @property
        def best_trial(self) -> Any:
            raise ValueError("no best trial")

        def optimize(self, objective, n_trials: int, callbacks):  # type: ignore[no-untyped-def]
            _ = objective, n_trials, callbacks
            return None

    monkeypatch.setattr(tuning.optuna, "create_study", lambda **kwargs: _FakeStudy())

    with pytest.raises(VeldraValidationError, match="without a completed trial"):
        tuning.run_tuning(
            config,
            frame,
            run_id="rid_edge",
            study_name="edge_no_trial",
            storage_url=f"sqlite:///{(tmp_path / 'unused.db').resolve()}",
            resume=False,
            output_dir=tmp_path / "tune_out",
        )
