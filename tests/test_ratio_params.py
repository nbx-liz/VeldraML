from __future__ import annotations

import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import regression


class _DummyBooster:
    best_iteration = 1

    def model_to_string(self) -> str:
        return "dummy"


def _xy(n_rows: int = 23) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame(
        {
            "x1": [float(i) for i in range(n_rows)],
            "x2": [float(i) * 0.5 for i in range(n_rows)],
        }
    )
    y = pd.Series([float(i) * 0.1 for i in range(n_rows)], name="target")
    return x, y


def test_ratio_params_are_resolved_into_absolute_values(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "train": {
                "min_data_in_leaf_ratio": 0.05,
                "min_data_in_bin_ratio": 0.01,
            },
        }
    )
    x, y = _xy(23)
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(regression.lgb, "train", _fake_train)
    regression._train_single_booster(x, y, x, y, cfg)

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["min_data_in_leaf"] == 2
    assert params["min_data_in_bin"] == 1


def test_ratio_params_are_not_injected_when_not_configured(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
        }
    )
    x, y = _xy(10)
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        return _DummyBooster()

    monkeypatch.setattr(regression.lgb, "train", _fake_train)
    regression._train_single_booster(x, y, x, y, cfg)

    params = captured["params"]
    assert isinstance(params, dict)
    assert "min_data_in_leaf" not in params
    assert "min_data_in_bin" not in params
