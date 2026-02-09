from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import tuning


def _config(task_type: str = "regression") -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": task_type},
            "data": {"path": "dummy.csv", "target": "target"},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
        }
    )


class _Trial:
    def suggest_int(self, name: str, low: int, high: int) -> int:
        _ = name, low, high
        return 3

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        _ = name, low, high, log
        return 0.3

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        _ = name
        return choices[0]


def test_objective_and_default_search_space_validation() -> None:
    assert tuning._objective_spec("regression") == ("rmse", "minimize")
    assert tuning._objective_spec("binary") == ("auc", "maximize")
    assert tuning._objective_spec("multiclass") == ("macro_f1", "maximize")
    with pytest.raises(VeldraValidationError):
        tuning._objective_spec("frontier")
    with pytest.raises(VeldraValidationError):
        tuning._default_search_space("regression", "unknown")
    assert tuning._default_search_space("regression", "standard")


def test_resolve_search_space_prefers_config_override() -> None:
    cfg = _config("regression")
    cfg.tuning.search_space = {"num_leaves": {"type": "int", "low": 10, "high": 20}}
    assert tuning._resolve_search_space(cfg) == cfg.tuning.search_space


def test_suggest_from_spec_variants_and_validation() -> None:
    trial = _Trial()
    assert tuning._suggest_from_spec(trial, "a", {"type": "int", "low": 1, "high": 3}) == 3
    assert tuning._suggest_from_spec(trial, "b", {"type": "float", "low": 0.1, "high": 1.0}) == 0.3
    assert tuning._suggest_from_spec(
        trial, "c", {"type": "categorical", "choices": ["x", "y"]}
    ) == "x"
    assert tuning._suggest_from_spec(trial, "d", [1, 2, 3]) == 1
    assert tuning._suggest_from_spec(trial, "e", 42) == 42

    with pytest.raises(VeldraValidationError):
        tuning._suggest_from_spec(trial, "bad", {"type": "categorical", "choices": []})
    with pytest.raises(VeldraValidationError):
        tuning._suggest_from_spec(trial, "bad", {"type": "unknown"})
    with pytest.raises(VeldraValidationError):
        tuning._suggest_from_spec(trial, "bad", [])


def test_score_for_task_validation_paths(monkeypatch) -> None:
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0.0, 1.0]})
    with pytest.raises(VeldraValidationError):
        tuning._score_for_task(_config("frontier"), frame, "rmse")

    def _fake_train_regression_with_cv(config, data):  # type: ignore[no-untyped-def]
        _ = config, data
        return SimpleNamespace(metrics={"mean": {}})

    monkeypatch.setattr(tuning, "train_regression_with_cv", _fake_train_regression_with_cv)
    with pytest.raises(VeldraValidationError):
        tuning._score_for_task(_config("regression"), frame, "rmse")


def test_run_tuning_rejects_unsupported_task() -> None:
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0.0, 1.0]})
    with pytest.raises(VeldraValidationError):
        tuning.run_tuning(_config("frontier"), frame)

