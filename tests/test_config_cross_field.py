from __future__ import annotations

import pytest

from veldra.config.models import RunConfig


def _base_payload(task_type: str) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": "dummy.csv", "target": "target"},
        "split": {"type": "kfold", "n_splits": 3, "seed": 7},
        "export": {"artifact_dir": "artifacts"},
    }
    if task_type in {"binary", "multiclass"}:
        payload["split"] = {"type": "stratified", "n_splits": 3, "seed": 7}
    if task_type == "frontier":
        payload["frontier"] = {"alpha": 0.9}
    return payload


def test_frontier_alpha_must_be_open_interval() -> None:
    payload = _base_payload("frontier")
    payload["frontier"] = {"alpha": 0.0}
    with pytest.raises(ValueError, match="frontier.alpha"):
        RunConfig.model_validate(payload)

    payload["frontier"] = {"alpha": 1.0}
    with pytest.raises(ValueError, match="frontier.alpha"):
        RunConfig.model_validate(payload)


def test_causal_drdid_requires_panel_fields() -> None:
    payload = _base_payload("regression")
    payload["causal"] = {
        "method": "dr_did",
        "treatment_col": "treatment",
        "design": "panel",
    }
    with pytest.raises(ValueError, match="causal.time_col"):
        RunConfig.model_validate(payload)


def test_tuning_objective_must_match_task_type() -> None:
    payload = _base_payload("regression")
    payload["tuning"] = {"enabled": True, "n_trials": 1, "objective": "auc"}
    with pytest.raises(ValueError, match="tuning.objective"):
        RunConfig.model_validate(payload)


def test_train_metrics_must_match_task_type() -> None:
    payload = _base_payload("regression")
    payload["train"] = {"metrics": ["rmse", "auc"]}
    with pytest.raises(ValueError, match="train.metrics"):
        RunConfig.model_validate(payload)


def test_tuning_metrics_candidates_must_match_task_type() -> None:
    payload = _base_payload("multiclass")
    payload["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "metrics_candidates": ["multi_logloss", "auc"],
    }
    with pytest.raises(ValueError, match="metrics_candidates"):
        RunConfig.model_validate(payload)
