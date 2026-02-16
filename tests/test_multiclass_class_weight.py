from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import multiclass


class _DummyBooster:
    best_iteration = 1

    def model_to_string(self) -> str:
        return "dummy"


def _config(
    *, auto_class_weight: bool = True, class_weight: dict[str, float] | None = None
) -> RunConfig:
    payload: dict = {
        "config_version": 1,
        "task": {"type": "multiclass"},
        "data": {"path": "dummy.csv", "target": "target"},
        "train": {"auto_class_weight": auto_class_weight},
    }
    if class_weight is not None:
        payload["train"]["class_weight"] = class_weight
    return RunConfig.model_validate(payload)


def _frame() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "x2": [1, 2, 3, 1, 2, 3]})
    y = pd.Series([0, 0, 0, 1, 2, 2], name="target")
    return x, y


def test_multiclass_auto_class_weight_sets_sample_weights(monkeypatch) -> None:
    config = _config(auto_class_weight=True)
    x, y = _frame()
    captured: dict = {}

    def _fake_dataset(data, label=None, weight=None, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return {"data": data, "label": label, "weight": weight}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["weight"] = kwargs["train_set"]["weight"]
        return _DummyBooster()

    monkeypatch.setattr(multiclass.lgb, "Dataset", _fake_dataset)
    monkeypatch.setattr(multiclass.lgb, "train", _fake_train)

    multiclass._train_single_booster(x, y, x, y, config, num_class=3)
    assert captured["weight"] is not None
    assert np.unique(captured["weight"]).size > 1


def test_multiclass_manual_class_weight_uses_given_weights(monkeypatch) -> None:
    config = _config(auto_class_weight=False, class_weight={"0": 1.0, "1": 2.0, "2": 3.0})
    x, y = _frame()
    captured: dict = {}

    def _fake_dataset(data, label=None, weight=None, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return {"data": data, "label": label, "weight": weight}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["weight"] = kwargs["train_set"]["weight"]
        return _DummyBooster()

    monkeypatch.setattr(multiclass.lgb, "Dataset", _fake_dataset)
    monkeypatch.setattr(multiclass.lgb, "train", _fake_train)

    multiclass._train_single_booster(x, y, x, y, config, num_class=3)
    assert captured["weight"].tolist() == [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]
