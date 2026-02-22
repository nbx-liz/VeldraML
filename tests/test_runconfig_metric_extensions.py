from __future__ import annotations

import pytest

from veldra.config.models import RunConfig


def _base_payload(task_type: str) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": "dummy.csv", "target": "target"},
        "split": {"type": "kfold", "n_splits": 3, "seed": 11},
        "export": {"artifact_dir": "artifacts"},
    }
    if task_type in {"binary", "multiclass"}:
        payload["split"] = {"type": "stratified", "n_splits": 3, "seed": 11}
    if task_type == "frontier":
        payload["frontier"] = {"alpha": 0.9}
    return payload


def test_train_metrics_accepts_task_specific_values() -> None:
    payload = _base_payload("regression")
    payload["train"] = {"metrics": ["rmse", "mae", "huber", "mape"]}
    config = RunConfig.model_validate(payload)
    assert config.train.metrics == ["rmse", "mae", "huber", "mape"]


def test_train_metrics_rejects_invalid_values() -> None:
    payload = _base_payload("binary")
    payload["train"] = {"metrics": ["auc", "rmse"]}
    with pytest.raises(ValueError, match="train.metrics"):
        RunConfig.model_validate(payload)


def test_tuning_metrics_candidates_accepts_multiclass_new_values() -> None:
    payload = _base_payload("multiclass")
    payload["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "metrics_candidates": ["multi_logloss", "multi_error"],
    }
    config = RunConfig.model_validate(payload)
    assert config.tuning.metrics_candidates == ["multi_logloss", "multi_error"]


def test_tuning_metrics_candidates_accepts_regression_huber() -> None:
    payload = _base_payload("regression")
    payload["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "metrics_candidates": ["rmse", "huber", "mae"],
    }
    config = RunConfig.model_validate(payload)
    assert config.tuning.metrics_candidates == ["rmse", "huber", "mae"]


def test_tuning_metrics_candidates_rejects_invalid_values() -> None:
    payload = _base_payload("regression")
    payload["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "metrics_candidates": ["rmse", "multi_error"],
    }
    with pytest.raises(ValueError, match="metrics_candidates"):
        RunConfig.model_validate(payload)


def test_tuning_metrics_candidates_rejects_mape_for_regression() -> None:
    payload = _base_payload("regression")
    payload["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "metrics_candidates": ["rmse", "mape"],
    }
    with pytest.raises(ValueError, match="metrics_candidates"):
        RunConfig.model_validate(payload)


def test_tuning_metrics_candidates_for_causal_follow_objective_set() -> None:
    payload = _base_payload("regression")
    payload["causal"] = {"method": "dr", "treatment_col": "treatment"}
    payload["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "objective": "dr_std_error",
        "metrics_candidates": ["dr_std_error", "dr_overlap_penalty"],
    }
    config = RunConfig.model_validate(payload)
    assert config.tuning.metrics_candidates == ["dr_std_error", "dr_overlap_penalty"]

    payload["tuning"]["metrics_candidates"] = ["dr_std_error", "rmse"]
    with pytest.raises(ValueError, match="metrics_candidates"):
        RunConfig.model_validate(payload)
