from __future__ import annotations

from pathlib import Path
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
    assert tuning._objective_spec("regression", None) == ("rmse", "minimize")
    assert tuning._objective_spec("binary", None) == ("auc", "maximize")
    assert tuning._objective_spec("multiclass", None) == ("macro_f1", "maximize")
    assert tuning._objective_spec("frontier", None) == ("pinball", "minimize")
    assert tuning._objective_spec("regression", "r2") == ("r2", "maximize")
    assert tuning._objective_spec("binary", "logloss") == ("logloss", "minimize")
    assert tuning._objective_spec("multiclass", "logloss") == ("logloss", "minimize")
    assert tuning._objective_spec("frontier", "pinball") == ("pinball", "minimize")
    with pytest.raises(VeldraValidationError):
        tuning._objective_spec("binary", "r2")
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

    def _fake_train_regression_with_cv(config, data):  # type: ignore[no-untyped-def]
        _ = config, data
        return SimpleNamespace(metrics={"mean": {}})

    def _fake_train_frontier_with_cv(config, data):  # type: ignore[no-untyped-def]
        _ = config, data
        return SimpleNamespace(metrics={"mean": {}})

    monkeypatch.setattr(tuning, "train_regression_with_cv", _fake_train_regression_with_cv)
    monkeypatch.setattr(tuning, "train_frontier_with_cv", _fake_train_frontier_with_cv)
    with pytest.raises(VeldraValidationError):
        tuning._score_for_task(_config("regression"), frame, "rmse")
    with pytest.raises(VeldraValidationError):
        tuning._score_for_task(_config("frontier"), frame, "pinball")


def test_score_for_task_frontier_uses_frontier_trainer(monkeypatch) -> None:
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0.0, 1.0]})

    def _fake_train_frontier_with_cv(config, data):  # type: ignore[no-untyped-def]
        _ = config, data
        return SimpleNamespace(metrics={"mean": {"pinball": 0.42}})

    monkeypatch.setattr(tuning, "train_frontier_with_cv", _fake_train_frontier_with_cv)
    assert tuning._score_for_task(_config("frontier"), frame, "pinball") == 0.42


def test_run_tuning_accepts_frontier_task(monkeypatch, tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0.0, 1.0]})

    def _fake_train_frontier_with_cv(config, data):  # type: ignore[no-untyped-def]
        _ = config, data
        return SimpleNamespace(metrics={"mean": {"pinball": 0.5}})

    monkeypatch.setattr(tuning, "train_frontier_with_cv", _fake_train_frontier_with_cv)

    out = tuning.run_tuning(
        _config("frontier"),
        frame,
        run_id="rid",
        study_name="frontier_internal",
        storage_url=f"sqlite:///{(tmp_path / 'study.db').resolve()}",
        resume=False,
        output_dir=Path(tmp_path),
    )
    assert out.metric_name == "pinball"
    assert out.direction == "minimize"


def test_build_study_name_is_deterministic() -> None:
    cfg = _config("regression")
    first = tuning.build_study_name(cfg)
    second = tuning.build_study_name(cfg)
    assert first == second
    assert first.startswith("regression_tune_")
