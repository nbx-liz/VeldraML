from __future__ import annotations

import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import regression


class _DummyBooster:
    best_iteration = 1

    def model_to_string(self) -> str:
        return "dummy"


def _xy() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.0, 1.2, 0.8, 1.1]})
    y = pd.Series([1.0, 1.1, 1.2, 1.3], name="target")
    return x, y


def test_feature_weights_are_injected_in_feature_order(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "train": {"feature_weights": {"x1": 2.0}},
        }
    )
    x, y = _xy()
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(regression.lgb, "train", _fake_train)
    regression._train_single_booster(x, y, x, y, cfg)

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["feature_weights"] == [2.0, 1.0]
    assert params["feature_pre_filter"] is False


def test_feature_weights_are_not_injected_when_not_configured(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
        }
    )
    x, y = _xy()
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(regression.lgb, "train", _fake_train)
    regression._train_single_booster(x, y, x, y, cfg)

    params = captured["params"]
    assert isinstance(params, dict)
    assert "feature_weights" not in params
    assert "feature_pre_filter" not in params
