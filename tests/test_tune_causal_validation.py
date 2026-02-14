from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import VeldraValidationError, tune


def test_causal_tune_rejects_invalid_objective_for_method(tmp_path) -> None:
    frame = pd.DataFrame(
        {"x1": [0.0, 0.5, 1.0, 1.5], "treatment": [0, 1, 0, 1], "outcome": [1.0, 2.0, 1.3, 2.5]}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "outcome"},
                "causal": {"method": "dr", "treatment_col": "treatment"},
                "tuning": {"enabled": True, "n_trials": 1, "objective": "drdid_std_error"},
            }
        )


def test_causal_tune_rejects_negative_causal_penalty_weight(tmp_path) -> None:
    frame = pd.DataFrame(
        {"x1": [0.0, 0.5, 1.0, 1.5], "treatment": [0, 1, 0, 1], "outcome": [1.0, 2.0, 1.3, 2.5]}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "outcome"},
                "causal": {"method": "dr", "treatment_col": "treatment"},
                "tuning": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "dr_overlap_penalty",
                    "causal_penalty_weight": -1.0,
                },
            }
        )


def test_causal_tune_rejects_non_positive_balance_threshold(tmp_path) -> None:
    frame = pd.DataFrame(
        {"x1": [0.0, 0.5, 1.0, 1.5], "treatment": [0, 1, 0, 1], "outcome": [1.0, 2.0, 1.3, 2.5]}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "outcome"},
                "causal": {"method": "dr", "treatment_col": "treatment"},
                "tuning": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "dr_balance_priority",
                    "causal_balance_threshold": 0.0,
                },
            }
        )
