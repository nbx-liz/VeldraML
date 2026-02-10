from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import tune
from veldra.api.exceptions import VeldraValidationError


def test_tune_frontier_rejects_stratified_split(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.5, 1.0, 1.5],
            "target": [0.2, 0.6, 1.1, 1.4],
        }
    )
    path = tmp_path / "frontier.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "frontier"},
                "data": {"path": str(path), "target": "target"},
                "split": {"type": "stratified", "n_splits": 2, "seed": 5},
                "tuning": {"enabled": True, "n_trials": 1},
            }
        )


def test_tune_frontier_requires_enabled_and_positive_trials(tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0], "target": [0.3, 1.3]})
    path = tmp_path / "frontier_small.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError, match="tuning.enabled"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "frontier"},
                "data": {"path": str(path), "target": "target"},
                "tuning": {"enabled": False, "n_trials": 1},
            }
        )

    with pytest.raises(VeldraValidationError, match="n_trials"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "frontier"},
                "data": {"path": str(path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 5},
                "tuning": {"enabled": True, "n_trials": 0},
            }
        )
