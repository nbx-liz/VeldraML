from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from veldra.config.models import RunConfig
from veldra.modeling import tuning


def _config(method: str, objective: str) -> RunConfig:
    payload: dict[str, object] = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "dummy.csv", "target": "outcome"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 3},
        "tuning": {
            "enabled": True,
            "n_trials": 1,
            "objective": objective,
            "causal_penalty_weight": 2.0,
            "causal_balance_threshold": 0.10,
        },
        "causal": {"method": method, "treatment_col": "treatment"},
    }
    if method == "dr_did":
        payload["causal"] = {
            "method": "dr_did",
            "treatment_col": "treatment",
            "design": "panel",
            "time_col": "time",
            "post_col": "post",
            "unit_id_col": "unit_id",
        }
    return RunConfig.model_validate(payload)


def _estimation(std_error: float, smd_weighted: float, overlap: float) -> SimpleNamespace:
    return SimpleNamespace(
        estimate=0.2,
        std_error=std_error,
        metrics={
            "overlap_metric": overlap,
            "smd_max_unweighted": 0.12,
            "smd_max_weighted": smd_weighted,
        },
    )


def test_balance_priority_uses_std_error_when_balanced(monkeypatch) -> None:
    cfg = _config("dr", "dr_balance_priority")
    monkeypatch.setattr(
        tuning,
        "run_dr_estimation",
        lambda *_args, **_kwargs: _estimation(std_error=0.33, smd_weighted=0.08, overlap=0.5),
    )
    value, components = tuning._score_for_task_with_components(
        cfg,
        pd.DataFrame({"x1": [0.0], "treatment": [0], "outcome": [0.0]}),
        "dr_balance_priority",
    )
    assert value == 0.33
    assert components["balance_violation"] == 0.0
    assert components["objective_stage"] == "balanced"


def test_balance_priority_uses_violation_priority_when_unbalanced(monkeypatch) -> None:
    cfg = _config("dr_did", "drdid_balance_priority")
    monkeypatch.setattr(
        tuning,
        "run_dr_did_estimation",
        lambda *_args, **_kwargs: _estimation(std_error=0.25, smd_weighted=0.15, overlap=0.6),
    )
    value, components = tuning._score_for_task_with_components(
        cfg,
        pd.DataFrame({"x1": [0.0], "treatment": [0], "outcome": [0.0]}),
        "drdid_balance_priority",
    )
    # violation = 0.15 - 0.10 = 0.05 ; penalty = 2.0 * 0.05 = 0.1
    assert value == 1_000_000.1
    assert components["balance_violation"] == pytest.approx(0.05)
    assert components["penalty"] == pytest.approx(0.1)
    assert components["objective_stage"] == "violated"
