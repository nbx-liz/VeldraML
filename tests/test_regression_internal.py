import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import regression


def _config_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "dummy.csv", "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
    }


def _build_config() -> RunConfig:
    return RunConfig.model_validate(_config_payload())


def _build_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "x2": [1.0, 1.2, 1.1, 0.9, 1.3, 1.4],
            "group": [0, 0, 1, 1, 1, 2],
            "ts": [1, 2, 3, 4, 5, 6],
            "target": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        }
    )


def test_build_feature_frame_validation_errors() -> None:
    config = _build_config()
    frame = _build_frame()

    with pytest.raises(VeldraValidationError):
        regression._build_feature_frame(config, frame.drop(columns=["target"]))
    with pytest.raises(VeldraValidationError):
        regression._build_feature_frame(config, frame.iloc[0:0])

    config.data.drop_cols = ["x1", "x2", "group", "ts"]
    with pytest.raises(VeldraValidationError):
        regression._build_feature_frame(config, frame)


def test_iter_cv_splits_validation_errors() -> None:
    config = _build_config()
    frame = _build_frame()
    x = frame[["x1", "x2"]]

    config.split.type = "group"  # type: ignore[assignment]
    config.split.group_col = None
    with pytest.raises(VeldraValidationError):
        regression._iter_cv_splits(config, frame, x)

    config.split.group_col = "missing_group"
    with pytest.raises(VeldraValidationError):
        regression._iter_cv_splits(config, frame, x)

    config.split.type = "timeseries"  # type: ignore[assignment]
    config.split.time_col = None
    with pytest.raises(VeldraValidationError):
        regression._iter_cv_splits(config, frame, x)

    config.split.type = "stratified"  # type: ignore[assignment]
    with pytest.raises(VeldraValidationError):
        regression._iter_cv_splits(config, frame, x)

    config.split.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraValidationError):
        regression._iter_cv_splits(config, frame, x)


def test_iter_cv_splits_group_and_timeseries_success() -> None:
    config = _build_config()
    frame = _build_frame()
    x = frame[["x1", "x2"]]

    config.split.type = "group"  # type: ignore[assignment]
    config.split.group_col = "group"
    group_splits = regression._iter_cv_splits(config, frame, x)
    assert group_splits

    config.split.type = "timeseries"  # type: ignore[assignment]
    config.split.time_col = "ts"
    ts_splits = regression._iter_cv_splits(config, frame, x)
    assert ts_splits


def test_train_regression_with_cv_early_validation() -> None:
    frame = _build_frame()

    non_reg_payload = _config_payload()
    non_reg_payload["task"] = {"type": "binary"}
    non_reg = RunConfig.model_validate(non_reg_payload)
    with pytest.raises(VeldraValidationError):
        regression.train_regression_with_cv(non_reg, frame)

    cfg = _build_config()
    cfg.data.path = None
    with pytest.raises(VeldraValidationError):
        regression.train_regression_with_cv(cfg, frame)


class _FakeBooster:
    def __init__(self, pred_value: float = 0.0) -> None:
        self.best_iteration = 1
        self._pred_value = pred_value

    def predict(self, x: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        _ = num_iteration
        return np.full(len(x), self._pred_value, dtype=float)

    def model_to_string(self) -> str:
        return "fake-model"


def test_train_regression_with_cv_rejects_empty_split(monkeypatch) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        regression,
        "_iter_cv_splits",
        lambda config, data, x: [(np.array([], dtype=int), np.array([0, 1], dtype=int))],
    )

    with pytest.raises(VeldraValidationError):
        regression.train_regression_with_cv(cfg, frame)


def test_train_regression_with_cv_rejects_nan_oof(monkeypatch) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        regression,
        "_iter_cv_splits",
        lambda config, data, x: [(np.array([0, 1], dtype=int), np.array([2, 3], dtype=int))],
    )
    monkeypatch.setattr(
        regression,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(pred_value=0.0),
    )

    with pytest.raises(VeldraValidationError):
        regression.train_regression_with_cv(cfg, frame)


def test_train_regression_with_cv_timeseries_path(monkeypatch) -> None:
    cfg = _build_config()
    cfg.split.type = "timeseries"  # type: ignore[assignment]
    cfg.split.time_col = "ts"
    frame = _build_frame().sample(frac=1.0, random_state=3).reset_index(drop=True)

    monkeypatch.setattr(
        regression,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(pred_value=2.4),
    )
    monkeypatch.setattr(
        regression,
        "_iter_cv_splits",
        lambda config, data, x: [
            (np.array([0, 1, 2, 3], dtype=int), np.array([4, 5], dtype=int)),
            (np.array([2, 3, 4, 5], dtype=int), np.array([0, 1], dtype=int)),
            (np.array([0, 1, 4, 5], dtype=int), np.array([2, 3], dtype=int)),
        ],
    )
    output = regression.train_regression_with_cv(cfg, frame)
    assert output.metrics["mean"]["rmse"] >= 0.0
