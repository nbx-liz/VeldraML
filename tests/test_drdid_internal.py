from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.causal import dr_did
from veldra.config.models import RunConfig


def _config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "outcome"},
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


def _panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "post": [0, 1, 0, 1],
            "treatment": [0, 0, 1, 1],
            "x": [0.2, 0.2, 0.8, 0.8],
            "outcome": [10.0, 11.0, 13.0, 15.0],
        }
    )


def test_to_binary_and_overlap_metric_edges() -> None:
    out = dr_did._to_binary(pd.Series([0, 1, 0, 1]), name="flag")
    assert out.dtype == int

    with pytest.raises(VeldraValidationError, match="must be binary"):
        dr_did._to_binary(pd.Series([0, np.nan]), name="flag")

    with pytest.raises(VeldraValidationError, match="exactly two binary values"):
        dr_did._to_binary(pd.Series([0, 1, 2]), name="flag")

    e_hat = np.array([0.2, 0.4, 0.8], dtype=float)
    treat = np.array([1, 1, 1], dtype=int)
    assert dr_did._overlap_metric(e_hat, treat) == 0.0


def test_base_validation_rejects_missing_and_non_numeric_outcome() -> None:
    cfg = _config()

    with pytest.raises(VeldraValidationError, match="Input data is empty"):
        dr_did._base_validation(cfg, pd.DataFrame())

    missing = _panel_frame().drop(columns=["post"])
    with pytest.raises(VeldraValidationError, match="missing required columns"):
        dr_did._base_validation(cfg, missing)

    bad_y = _panel_frame().copy()
    bad_y["outcome"] = ["a", "b", "c", "d"]
    with pytest.raises(VeldraValidationError, match="Outcome values must be numeric"):
        dr_did._base_validation(cfg, bad_y)


def test_dr_config_from_drdid_requires_causal() -> None:
    cfg = _config()
    cfg.causal = None
    with pytest.raises(VeldraValidationError, match="causal config is required"):
        dr_did._dr_config_from_drdid(cfg)


def test_panel_to_pseudo_frame_validation_paths(monkeypatch) -> None:
    cfg = _config()

    no_post = _panel_frame().loc[lambda d: d["post"] == 0].reset_index(drop=True)
    with pytest.raises(VeldraValidationError, match="requires both pre and post"):
        dr_did._panel_to_pseudo_frame(cfg, no_post)

    duplicate_pre = pd.concat(
        [_panel_frame(), _panel_frame().iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(VeldraValidationError, match="exactly one pre and one post"):
        dr_did._panel_to_pseudo_frame(cfg, duplicate_pre)

    unstable_treat = _panel_frame().copy()
    unstable_mask = (unstable_treat["unit_id"] == 2) & (unstable_treat["post"] == 1)
    unstable_treat.loc[unstable_mask, "treatment"] = 0
    with pytest.raises(VeldraValidationError, match="treatment to be stable"):
        dr_did._panel_to_pseudo_frame(cfg, unstable_treat)

    cfg.data.drop_cols = ["x"]
    with pytest.raises(VeldraValidationError, match="No feature columns remain"):
        dr_did._panel_to_pseudo_frame(cfg, _panel_frame())

    cfg = _config()
    monkeypatch.setattr(dr_did.pd, "get_dummies", lambda *_args, **_kwargs: pd.DataFrame())
    with pytest.raises(
        VeldraValidationError,
        match="No usable feature columns remain after encoding",
    ):
        dr_did._panel_to_pseudo_frame(cfg, _panel_frame())


def test_repeated_cs_requires_pre_and_post() -> None:
    cfg = _config()
    assert cfg.causal is not None
    cfg.causal.design = "repeated_cross_section"
    cfg.causal.unit_id_col = None

    frame = _panel_frame().copy()
    frame["post"] = 1
    with pytest.raises(VeldraValidationError, match="requires both pre and post rows"):
        dr_did._repeated_cs_to_pseudo_frame(cfg, frame)
