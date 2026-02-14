import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import multiclass


def _build_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 7},
        }
    )


def _build_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 2.1, 2.2, 2.3, 4.1, 4.2, 4.3],
            "x2": [1.0, 1.1, 0.9, 0.1, 0.0, 0.2, -1.0, -0.8, -0.9],
            "group": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "ts": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "target": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        }
    )


def test_build_feature_frame_validation_errors() -> None:
    config = _build_config()
    frame = _build_frame()

    with pytest.raises(VeldraValidationError):
        multiclass._build_feature_frame(config, frame.drop(columns=["target"]))
    with pytest.raises(VeldraValidationError):
        multiclass._build_feature_frame(config, frame.iloc[0:0])

    bad = frame.copy()
    bad.loc[0, "target"] = np.nan
    with pytest.raises(VeldraValidationError):
        multiclass._build_feature_frame(config, bad)

    two_class = frame.copy()
    two_class["target"] = ["a"] * 5 + ["b"] * 4
    with pytest.raises(VeldraValidationError):
        multiclass._build_feature_frame(config, two_class)

    config.data.drop_cols = ["x1", "x2", "group", "ts"]
    with pytest.raises(VeldraValidationError):
        multiclass._build_feature_frame(config, frame)


def test_multiclass_to_python_scalar_and_encoding_failure(monkeypatch) -> None:
    assert multiclass._to_python_scalar(np.int64(5)) == 5

    cfg = _build_config()
    frame = _build_frame()
    monkeypatch.setattr(multiclass, "_to_python_scalar", lambda value: "same")
    with pytest.raises(VeldraValidationError, match="Failed to encode"):
        multiclass._build_feature_frame(cfg, frame)


def test_normalize_proba_validation_errors() -> None:
    with pytest.raises(VeldraValidationError):
        multiclass._normalize_proba(np.array([0.1, 0.2, 0.3]), n_rows=2, num_class=3)
    with pytest.raises(VeldraValidationError):
        multiclass._normalize_proba(np.ones((4, 2)), n_rows=4, num_class=3)


def test_train_multiclass_with_cv_early_validation() -> None:
    frame = _build_frame()

    non_mc_payload = {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": "dummy.csv", "target": "target"},
        "split": {"type": "stratified", "n_splits": 3, "seed": 7},
    }
    non_mc = RunConfig.model_validate(non_mc_payload)
    with pytest.raises(VeldraValidationError):
        multiclass.train_multiclass_with_cv(non_mc, frame)

    cfg = _build_config()
    cfg.data.path = None
    with pytest.raises(VeldraValidationError):
        multiclass.train_multiclass_with_cv(cfg, frame)


class _FakeBooster:
    def __init__(self, n_class: int, pred_row: list[float]) -> None:
        self.best_iteration = 1
        self._n_class = n_class
        self._pred_row = np.asarray(pred_row, dtype=float)

    def predict(self, x: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        _ = num_iteration
        return np.tile(self._pred_row, (len(x), 1)).reshape(len(x) * self._n_class)

    def model_to_string(self) -> str:
        return "fake-model"


def test_train_multiclass_with_cv_rejects_empty_split(monkeypatch) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        multiclass,
        "iter_cv_splits",
        lambda config, data, x, y=None: [
            (np.array([], dtype=int), np.array([0, 1], dtype=int))
        ],
    )

    with pytest.raises(VeldraValidationError):
        multiclass.train_multiclass_with_cv(cfg, frame)


def test_train_multiclass_with_cv_rejects_nan_oof(monkeypatch) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        multiclass,
        "iter_cv_splits",
        lambda config, data, x, y=None: [
            (np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([6, 7], dtype=int))
        ],
    )
    monkeypatch.setattr(
        multiclass,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(n_class=3, pred_row=[0.1, 0.2, 0.7]),
    )

    with pytest.raises(VeldraValidationError):
        multiclass.train_multiclass_with_cv(cfg, frame)


def test_train_multiclass_with_cv_timeseries_path(monkeypatch) -> None:
    cfg = _build_config()
    cfg.split.type = "timeseries"  # type: ignore[assignment]
    cfg.split.time_col = "ts"
    frame = _build_frame().sample(frac=1.0, random_state=5).reset_index(drop=True)

    monkeypatch.setattr(
        multiclass,
        "iter_cv_splits",
        lambda config, data, x, y=None: [
            (np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([6, 7, 8], dtype=int)),
            (np.array([3, 4, 5, 6, 7, 8], dtype=int), np.array([0, 1, 2], dtype=int)),
            (np.array([0, 1, 2, 6, 7, 8], dtype=int), np.array([3, 4, 5], dtype=int)),
        ],
    )
    monkeypatch.setattr(
        multiclass,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(n_class=3, pred_row=[0.2, 0.3, 0.5]),
    )

    output = multiclass.train_multiclass_with_cv(cfg, frame)
    assert output.metrics["mean"]["accuracy"] >= 0.0
