from __future__ import annotations

import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import regression


class _DummyBooster:
    best_iteration = 1

    def model_to_string(self) -> str:
        return "dummy"


def _xy() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.0, 1.1, 0.9, 1.2]})
    y = pd.Series([1.0, 1.1, 1.2, 1.3], name="target")
    return x, y


def test_auto_num_leaves_injected_into_train_params(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "train": {
                "auto_num_leaves": True,
                "num_leaves_ratio": 0.5,
                "lgb_params": {"max_depth": 5},
            },
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
    assert params["num_leaves"] == 16


def test_auto_num_leaves_is_not_injected_by_default(monkeypatch) -> None:
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
    assert "num_leaves" not in params
