from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import VeldraValidationError, estimate_dr


def _binary_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "treatment": [0, 0, 1, 1, 0, 0, 1, 1],
            "x": [0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4],
            "outcome": [0, 1, 0, 1, 0, 0, 1, 1],
        }
    )


def _binary_repeated_cs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0, 0, 0, 1, 1, 1],
            "post": [0, 0, 0, 1, 1, 1],
            "treatment": [0, 1, 0, 1, 0, 1],
            "x": [0.1, 0.5, 0.3, 0.4, 0.2, 0.6],
            "outcome": [0, 1, 0, 1, 0, 1],
        }
    )


def _base_binary_panel_config(path: str, artifact_dir: str) -> dict:
    return {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": path, "target": "outcome"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
        "causal": {
            "method": "dr_did",
            "treatment_col": "treatment",
            "design": "panel",
            "time_col": "time",
            "post_col": "post",
            "unit_id_col": "unit_id",
        },
        "export": {"artifact_dir": artifact_dir},
    }


def test_drdid_binary_rejects_non_binary_outcome(tmp_path) -> None:
    frame = _binary_panel_frame()
    frame.loc[0, "outcome"] = 2
    path = tmp_path / "bad_outcome.csv"
    frame.to_csv(path, index=False)
    config = _base_binary_panel_config(str(path), str(tmp_path))

    with pytest.raises(
        VeldraValidationError, match="outcome must contain exactly two binary values"
    ):
        estimate_dr(config)


def test_drdid_binary_rejects_non_binary_treatment(tmp_path) -> None:
    frame = _binary_panel_frame()
    frame.loc[0, "treatment"] = 2
    path = tmp_path / "bad_treatment.csv"
    frame.to_csv(path, index=False)
    config = _base_binary_panel_config(str(path), str(tmp_path))

    with pytest.raises(
        VeldraValidationError, match="treatment must contain exactly two binary values"
    ):
        estimate_dr(config)


def test_drdid_binary_rejects_non_binary_post(tmp_path) -> None:
    frame = _binary_repeated_cs_frame()
    frame.loc[0, "post"] = 2
    path = tmp_path / "bad_post.csv"
    frame.to_csv(path, index=False)
    config = {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": str(path), "target": "outcome"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
        "causal": {
            "method": "dr_did",
            "treatment_col": "treatment",
            "design": "repeated_cross_section",
            "time_col": "time",
            "post_col": "post",
        },
        "export": {"artifact_dir": str(tmp_path)},
    }

    with pytest.raises(VeldraValidationError, match="post must contain exactly two binary values"):
        estimate_dr(config)


def test_drdid_binary_panel_requires_unit_id(tmp_path) -> None:
    frame = _binary_panel_frame()
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)
    config = _base_binary_panel_config(str(path), str(tmp_path))
    config["causal"].pop("unit_id_col")

    with pytest.raises(VeldraValidationError, match="Invalid RunConfig"):
        estimate_dr(config)


def test_drdid_binary_rejects_non_att_estimand(tmp_path) -> None:
    frame = _binary_panel_frame()
    path = tmp_path / "panel_estimand.csv"
    frame.to_csv(path, index=False)
    config = _base_binary_panel_config(str(path), str(tmp_path))
    config["causal"]["estimand"] = "ate"

    with pytest.raises(
        VeldraValidationError,
        match="causal.estimand must be 'att' when causal.method='dr_did'",
    ):
        estimate_dr(config)
