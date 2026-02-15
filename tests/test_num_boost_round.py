from __future__ import annotations

import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import binary, frontier, multiclass, regression


class _DummyBooster:
    best_iteration = 1

    def model_to_string(self) -> str:
        return "dummy"


def _base_payload(task_type: str, rounds: int) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": "dummy.csv", "target": "target"},
        "train": {"num_boost_round": rounds},
    }
    if task_type == "frontier":
        payload["frontier"] = {"alpha": 0.9}
    return payload


def _xy() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.0, 1.2, 0.8, 1.1]})
    y = pd.Series([0.0, 1.0, 0.0, 1.0], name="target")
    return x, y


def test_num_boost_round_for_regression(monkeypatch) -> None:
    cfg = RunConfig.model_validate(_base_payload("regression", 77))
    x, y = _xy()
    captured: dict = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["rounds"] = kwargs["num_boost_round"]
        return _DummyBooster()

    monkeypatch.setattr(regression.lgb, "train", _fake_train)
    regression._train_single_booster(x, y, x, y, cfg)
    assert captured["rounds"] == 77


def test_num_boost_round_for_binary(monkeypatch) -> None:
    cfg = RunConfig.model_validate(_base_payload("binary", 55))
    x, y = _xy()
    y = y.astype(int)
    captured: dict = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["rounds"] = kwargs["num_boost_round"]
        return _DummyBooster()

    monkeypatch.setattr(binary.lgb, "train", _fake_train)
    binary._train_single_booster(x, y, x, y, cfg)
    assert captured["rounds"] == 55


def test_num_boost_round_for_multiclass(monkeypatch) -> None:
    cfg = RunConfig.model_validate(_base_payload("multiclass", 33))
    x, y = _xy()
    y = pd.Series([0, 1, 2, 2], name="target")
    captured: dict = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["rounds"] = kwargs["num_boost_round"]
        return _DummyBooster()

    monkeypatch.setattr(multiclass.lgb, "train", _fake_train)
    multiclass._train_single_booster(x, y, x, y, cfg, num_class=3)
    assert captured["rounds"] == 33


def test_num_boost_round_for_frontier(monkeypatch) -> None:
    cfg = RunConfig.model_validate(_base_payload("frontier", 21))
    x, y = _xy()
    captured: dict = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["rounds"] = kwargs["num_boost_round"]
        return _DummyBooster()

    monkeypatch.setattr(frontier.lgb, "train", _fake_train)
    frontier._train_single_booster(x, y, x, y, cfg)
    assert captured["rounds"] == 21
