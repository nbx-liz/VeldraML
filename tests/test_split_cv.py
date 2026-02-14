from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.split import iter_cv_splits


def _binary_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 2.1, 0.2, 2.2, 0.3, 2.3, 0.4, 2.4],
            "x2": [1.0, 0.1, 1.1, 0.0, 0.9, 0.2, 1.2, 0.3],
            "group": [0, 0, 0, 1, 1, 1, 2, 2],
            "ts": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    x = frame[["x1", "x2"]]
    y = frame["target"]
    return frame, x, y


def test_iter_cv_splits_stratified_requires_y(config_payload) -> None:
    cfg = RunConfig.model_validate(config_payload("binary", split={"type": "stratified"}))
    frame, x, _ = _binary_data()
    with pytest.raises(VeldraValidationError, match="requires target labels"):
        iter_cv_splits(cfg, frame, x, None)


def test_iter_cv_splits_group_validation(config_payload) -> None:
    cfg = RunConfig.model_validate(
        config_payload(
            "regression",
            split={"type": "group", "group_col": "group", "n_splits": 2, "seed": 7},
        )
    )
    cfg.split.group_col = None
    frame, x, _ = _binary_data()
    with pytest.raises(VeldraValidationError, match="group_col is required"):
        iter_cv_splits(cfg, frame, x)

    cfg2 = RunConfig.model_validate(
        config_payload(
            "regression",
            split={"type": "group", "group_col": "missing", "n_splits": 2, "seed": 7},
        )
    )
    with pytest.raises(VeldraValidationError, match="Group column 'missing'"):
        iter_cv_splits(cfg2, frame, x)


def test_iter_cv_splits_success_paths(config_payload) -> None:
    frame, x, y = _binary_data()

    kfold_cfg = RunConfig.model_validate(config_payload("regression"))
    assert iter_cv_splits(kfold_cfg, frame, x)

    strat_cfg = RunConfig.model_validate(config_payload("binary"))
    assert iter_cv_splits(strat_cfg, frame, x, y)

    group_cfg = RunConfig.model_validate(
        config_payload(
            "regression",
            split={"type": "group", "group_col": "group", "n_splits": 2, "seed": 7},
        )
    )
    assert iter_cv_splits(group_cfg, frame, x)


def test_iter_cv_splits_timeseries_params(monkeypatch, config_payload) -> None:
    cfg = RunConfig.model_validate(
        config_payload(
            "regression",
            split={
                "type": "timeseries",
                "time_col": "ts",
                "timeseries_mode": "blocked",
                "test_size": 2,
                "gap": 1,
                "embargo": 2,
                "train_size": 3,
                "n_splits": 2,
                "seed": 7,
            },
        )
    )
    frame, x, _ = _binary_data()
    captured: dict[str, int | str | None] = {}

    class _FakeSplitter:
        def __init__(
            self,
            n_splits: int,
            test_size: int | None,
            gap: int,
            embargo: int,
            mode: str,
            train_size: int | None,
        ) -> None:
            captured["n_splits"] = n_splits
            captured["test_size"] = test_size
            captured["gap"] = gap
            captured["embargo"] = embargo
            captured["mode"] = mode
            captured["train_size"] = train_size

        def split(self, data: int):
            _ = data
            return [(np.array([0, 1, 2], dtype=int), np.array([3, 4], dtype=int))]

    monkeypatch.setattr("veldra.split.cv.TimeSeriesSplitter", _FakeSplitter)
    splits = iter_cv_splits(cfg, frame, x)
    assert splits
    assert captured == {
        "n_splits": cfg.split.n_splits,
        "test_size": 2,
        "gap": 1,
        "embargo": 2,
        "mode": "blocked",
        "train_size": 3,
    }
