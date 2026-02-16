from __future__ import annotations

import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling.utils import (
    resolve_auto_num_leaves,
    resolve_feature_weights,
    resolve_ratio_params,
)


def _cfg(train: dict | None = None) -> RunConfig:
    payload: dict = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "dummy.csv", "target": "target"},
    }
    if train is not None:
        payload["train"] = train
    return RunConfig.model_validate(payload)


def test_resolve_auto_num_leaves_from_max_depth() -> None:
    cfg = _cfg(
        {
            "auto_num_leaves": True,
            "num_leaves_ratio": 0.5,
            "lgb_params": {"max_depth": 5},
        }
    )
    assert resolve_auto_num_leaves(cfg) == 16


def test_resolve_auto_num_leaves_unlimited_depth_upper_cap() -> None:
    cfg = _cfg({"auto_num_leaves": True, "num_leaves_ratio": 1.0, "lgb_params": {"max_depth": -1}})
    assert resolve_auto_num_leaves(cfg) == 131072


def test_resolve_ratio_params_uses_row_count() -> None:
    cfg = _cfg({"min_data_in_leaf_ratio": 0.01, "min_data_in_bin_ratio": 0.001})
    resolved = resolve_ratio_params(cfg, n_rows=123)
    assert resolved["min_data_in_leaf"] == 2
    assert resolved["min_data_in_bin"] == 1


def test_resolve_feature_weights_in_feature_order() -> None:
    cfg = _cfg({"feature_weights": {"x1": 2.0}})
    assert resolve_feature_weights(cfg, ["x1", "x2"]) == [2.0, 1.0]


def test_resolve_feature_weights_rejects_unknown_columns() -> None:
    cfg = _cfg({"feature_weights": {"missing": 2.0}})
    with pytest.raises(VeldraValidationError, match="unknown features"):
        resolve_feature_weights(cfg, ["x1", "x2"])
