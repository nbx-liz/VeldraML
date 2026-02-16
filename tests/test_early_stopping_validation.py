from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import regression
from veldra.modeling.utils import split_for_early_stopping


def _regression_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 7},
            "train": {"early_stopping_rounds": 10, "early_stopping_validation_fraction": 0.25},
        }
    )


def test_split_for_early_stopping_disabled_returns_identity() -> None:
    cfg = _regression_config()
    cfg.train.early_stopping_rounds = None
    x = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    y = pd.Series([1.0, 2.0, 3.0])
    out = split_for_early_stopping(x, y, cfg)
    assert out[0].equals(x)
    assert out[1].equals(x)
    assert out[2].equals(y)
    assert out[3].equals(y)


def test_split_for_early_stopping_timeseries_uses_tail() -> None:
    cfg = _regression_config()
    cfg.split.type = "timeseries"  # type: ignore[assignment]
    cfg.split.time_col = "ts"
    cfg.train.early_stopping_validation_fraction = 0.4
    x = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "ts": [1, 2, 3, 4, 5]})
    y = pd.Series([10, 11, 12, 13, 14])
    x_train, x_valid, y_train, y_valid = split_for_early_stopping(x, y, cfg)
    assert x_train.index.tolist() == [0, 1, 2]
    assert x_valid.index.tolist() == [3, 4]
    assert y_train.index.tolist() == [0, 1, 2]
    assert y_valid.index.tolist() == [3, 4]


def test_split_for_early_stopping_binary_prefers_stratified() -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 7},
            "train": {"early_stopping_rounds": 10, "early_stopping_validation_fraction": 0.3},
        }
    )
    x = pd.DataFrame({"x1": np.arange(20), "x2": np.arange(20) * 0.5})
    y = pd.Series(([0, 1] * 10))
    _x_train, x_valid, _y_train, y_valid = split_for_early_stopping(x, y, cfg)
    assert y_valid.nunique() == 2
    assert len(x_valid) > 0


def test_cv_training_uses_es_split_not_oof_valid(monkeypatch) -> None:
    cfg = _regression_config()
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x2": [1.0, 1.2, 1.1, 0.9, 1.3, 1.4],
            "target": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        }
    )
    fold_valids = [
        np.array([4, 5], dtype=int),
        np.array([0, 1], dtype=int),
        np.array([2, 3], dtype=int),
    ]
    splits = [
        (np.array([0, 1, 2, 3], dtype=int), fold_valids[0]),
        (np.array([2, 3, 4, 5], dtype=int), fold_valids[1]),
        (np.array([0, 1, 4, 5], dtype=int), fold_valids[2]),
    ]
    monkeypatch.setattr(regression, "iter_cv_splits", lambda *_args, **_kwargs: splits)

    es_valid_indices: list[list[int]] = []

    def _fake_es_split(x, y, _config):  # type: ignore[no-untyped-def]
        es_valid_indices.append(x.iloc[[-1]].index.tolist())
        return x.iloc[:-1], x.iloc[[-1]], y.iloc[:-1], y.iloc[[-1]]

    class _FakeBooster:
        best_iteration = 1

        def predict(self, x, num_iteration=None):  # type: ignore[no-untyped-def]
            _ = num_iteration
            return np.full(len(x), 2.5, dtype=float)

        def model_to_string(self) -> str:
            return "fake"

    monkeypatch.setattr(regression, "split_for_early_stopping", _fake_es_split)
    monkeypatch.setattr(regression, "_train_single_booster", lambda **_kwargs: _FakeBooster())

    regression.train_regression_with_cv(cfg, frame)
    for es_valid, fold_valid in zip(es_valid_indices[:3], fold_valids):
        assert set(es_valid).isdisjoint(set(fold_valid.tolist()))


def test_training_history_records_best_iteration_contract(monkeypatch) -> None:
    cfg = _regression_config()
    cfg.train.num_boost_round = 20

    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x2": [1.0, 1.2, 1.1, 0.9, 1.3, 1.4],
            "target": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        }
    )
    splits = [
        (np.array([0, 1, 2], dtype=int), np.array([3, 4, 5], dtype=int)),
        (np.array([3, 4, 5], dtype=int), np.array([0, 1, 2], dtype=int)),
    ]
    monkeypatch.setattr(regression, "iter_cv_splits", lambda *_args, **_kwargs: splits)
    monkeypatch.setattr(
        regression,
        "split_for_early_stopping",
        lambda x, y, _cfg: (x, x, y, y),
    )

    captured_rounds: list[int] = []

    class _FakeBooster:
        best_iteration = 4

        def current_iteration(self) -> int:
            return 9

        def predict(self, x, num_iteration=None):  # type: ignore[no-untyped-def]
            _ = num_iteration
            return np.full(len(x), 2.5, dtype=float)

        def model_to_string(self) -> str:
            return "fake"

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured_rounds.append(int(kwargs["num_boost_round"]))
        return _FakeBooster()

    monkeypatch.setattr(regression.lgb, "train", _fake_train)

    output = regression.train_regression_with_cv(cfg, frame)

    assert captured_rounds == [20, 20, 20]
    final_history = output.training_history["final_model"]
    assert final_history["best_iteration"] == 4
    assert final_history["best_iteration"] < cfg.train.num_boost_round
