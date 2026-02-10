from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import tune
from veldra.api.exceptions import VeldraValidationError


def _frontier_data(tmp_path) -> str:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [1.0, 1.5, 2.0, 2.5],
            "target": [1.2, 1.6, 2.2, 3.1],
        }
    )
    path = tmp_path / "frontier.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_non_frontier_rejects_frontier_coverage_tuning_fields(tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0], "target": [0.0, 1.0]})
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "target"},
                "tuning": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "rmse",
                    "coverage_target": 0.9,
                },
            }
        )


def test_frontier_rejects_invalid_coverage_target(tmp_path) -> None:
    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "frontier"},
                "data": {"path": _frontier_data(tmp_path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 1},
                "tuning": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "pinball_coverage_penalty",
                    "coverage_target": 1.2,
                },
            }
        )


def test_frontier_rejects_negative_tolerance_and_weight(tmp_path) -> None:
    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "frontier"},
                "data": {"path": _frontier_data(tmp_path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 1},
                "tuning": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "pinball_coverage_penalty",
                    "coverage_tolerance": -0.1,
                },
            }
        )

    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "frontier"},
                "data": {"path": _frontier_data(tmp_path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 1},
                "tuning": {
                    "enabled": True,
                    "n_trials": 1,
                    "objective": "pinball_coverage_penalty",
                    "penalty_weight": -1.0,
                },
            }
        )
