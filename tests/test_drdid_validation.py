from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import VeldraValidationError, estimate_dr


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "post": [0, 1, 0, 1],
            "treatment": [0, 0, 1, 1],
            "x": [0.1, 0.2, 0.3, 0.4],
            "outcome": [10.0, 11.0, 12.0, 14.0],
        }
    )


def test_drdid_requires_design_specific_columns(tmp_path) -> None:
    frame = _base_frame()
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        estimate_dr(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "outcome"},
                "causal": {
                    "method": "dr_did",
                    "treatment_col": "treatment",
                    "design": "panel",
                    "time_col": "time",
                    "post_col": "post",
                },
            }
        )


def test_drdid_rejects_binary_task(tmp_path) -> None:
    frame = _base_frame()
    frame["outcome"] = [0, 0, 1, 1]
    path = tmp_path / "panel_binary.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        estimate_dr(
            {
                "config_version": 1,
                "task": {"type": "binary"},
                "data": {"path": str(path), "target": "outcome"},
                "causal": {
                    "method": "dr_did",
                    "treatment_col": "treatment",
                    "design": "panel",
                    "time_col": "time",
                    "post_col": "post",
                    "unit_id_col": "unit_id",
                },
            }
        )

