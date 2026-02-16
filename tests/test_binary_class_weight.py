from __future__ import annotations

import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import binary


class _DummyBooster:
    best_iteration = 1

    def model_to_string(self) -> str:
        return "dummy"


def _config(
    *, auto_class_weight: bool = True, class_weight: dict[str, float] | None = None
) -> RunConfig:
    payload: dict = {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": "dummy.csv", "target": "target"},
        "postprocess": {"calibration": "platt"},
        "train": {"auto_class_weight": auto_class_weight},
    }
    if class_weight is not None:
        payload["train"]["class_weight"] = class_weight
    return RunConfig.model_validate(payload)


def _frame() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4, 0.5], "x2": [1, 1, 2, 2, 3]})
    y = pd.Series([0, 0, 0, 1, 1], name="target")
    return x, y


def test_binary_auto_class_weight_sets_is_unbalance(monkeypatch) -> None:
    config = _config(auto_class_weight=True)
    x, y = _frame()
    captured: dict = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(binary.lgb, "train", _fake_train)

    binary._train_single_booster(x, y, x, y, config)
    assert captured["params"]["is_unbalance"] is True


def test_binary_manual_class_weight_sets_scale_pos_weight(monkeypatch) -> None:
    config = _config(auto_class_weight=False, class_weight={"0": 1.0, "1": 2.0})
    x, y = _frame()
    captured: dict = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(binary.lgb, "train", _fake_train)

    binary._train_single_booster(x, y, x, y, config)
    expected = (3 * 2.0) / (2 * 1.0)
    assert captured["params"]["scale_pos_weight"] == expected
