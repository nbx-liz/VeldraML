from __future__ import annotations

import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import binary


class _DummyBooster:
    best_iteration = 1

    def model_to_string(self) -> str:
        return "dummy"


def _xy() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.1, 1.0, 0.9, 1.2]})
    y = pd.Series([0, 1, 0, 1], name="target")
    return x, y


def test_binary_objective_prefers_user_override(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
            "train": {"lgb_params": {"objective": "cross_entropy"}},
        }
    )
    x, y = _xy()
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(binary.lgb, "train", _fake_train)
    binary._train_single_booster(x, y, x, y, cfg)

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["objective"] == "cross_entropy"


def test_binary_objective_uses_default_when_not_overridden(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
        }
    )
    x, y = _xy()
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(binary.lgb, "train", _fake_train)
    binary._train_single_booster(x, y, x, y, cfg)

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["objective"] == "binary"
