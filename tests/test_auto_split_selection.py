from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.config.models import RunConfig
from veldra.split.cv import iter_cv_splits


def _frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x2": [1.0, 1.2, 0.9, 0.8, 1.1, 1.3],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    return frame, frame[["x1", "x2"]], frame["target"]


def test_binary_kfold_is_auto_stratified(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 7},
        }
    )
    frame, x, y = _frame()
    called: dict[str, bool] = {}

    class _FakeStratifiedKFold:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            _ = kwargs
            called["stratified"] = True

        def split(self, x_data, y_data):  # type: ignore[no-untyped-def]
            _ = x_data, y_data
            return [(np.array([0, 1, 2, 3]), np.array([4, 5]))]

    class _FakeKFold:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            _ = kwargs
            called["kfold"] = True

        def split(self, x_data):  # type: ignore[no-untyped-def]
            _ = x_data
            return [(np.array([0, 1, 2, 3]), np.array([4, 5]))]

    monkeypatch.setattr("veldra.split.cv.StratifiedKFold", _FakeStratifiedKFold)
    monkeypatch.setattr("veldra.split.cv.KFold", _FakeKFold)

    splits = iter_cv_splits(cfg, frame, x, y)
    assert splits
    assert called.get("stratified") is True
    assert called.get("kfold") is not True


def test_regression_kfold_keeps_kfold(monkeypatch) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 7},
        }
    )
    frame, x, _y = _frame()
    called: dict[str, bool] = {}

    class _FakeKFold:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            _ = kwargs
            called["kfold"] = True

        def split(self, x_data):  # type: ignore[no-untyped-def]
            _ = x_data
            return [(np.array([0, 1, 2, 3]), np.array([4, 5]))]

    monkeypatch.setattr("veldra.split.cv.KFold", _FakeKFold)

    splits = iter_cv_splits(cfg, frame, x)
    assert splits
    assert called.get("kfold") is True
