from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from veldra.api import tune
from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError
from veldra.modeling import tuning as tuning_module


def _binary_frame(rows: int = 24, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = ((x1 - x2) > np.median(x1 - x2)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_requires_enabled_flag(tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0], "target": [0.0, 1.0]})
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(VeldraValidationError, match="tuning.enabled"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "target"},
                "tuning": {"enabled": False, "n_trials": 1},
            }
        )


def test_tune_frontier_is_not_implemented(tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0], "target": [0.0, 1.0]})
    path = tmp_path / "frontier.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(VeldraNotImplementedError):
        tune(
            {
                "config_version": 1,
                "task": {"type": "frontier"},
                "data": {"path": str(path), "target": "target"},
                "tuning": {"enabled": True, "n_trials": 1},
            }
        )


def test_tune_requires_path_and_positive_trials() -> None:
    with pytest.raises(VeldraValidationError, match="data.path"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"target": "target"},
                "tuning": {"enabled": True, "n_trials": 1},
            }
        )
    with pytest.raises(VeldraValidationError, match="n_trials"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": "dummy.csv", "target": "target"},
                "tuning": {"enabled": True, "n_trials": 0},
            }
        )


def test_binary_tune_forces_threshold_optimization_off(monkeypatch, tmp_path) -> None:
    frame = _binary_frame()
    path = tmp_path / "bin.csv"
    frame.to_csv(path, index=False)
    seen: dict[str, object] = {}

    def _fake_train_binary_with_cv(config, data):  # type: ignore[no-untyped-def]
        _ = data
        seen["enabled"] = (
            config.postprocess.threshold_optimization.enabled
            if config.postprocess.threshold_optimization is not None
            else None
        )
        return SimpleNamespace(metrics={"mean": {"auc": 0.75}})

    monkeypatch.setattr(tuning_module, "train_binary_with_cv", _fake_train_binary_with_cv)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 1},
            "postprocess": {"threshold_optimization": {"enabled": True, "objective": "f1"}},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert result.best_score == 0.75
    assert seen["enabled"] is False

