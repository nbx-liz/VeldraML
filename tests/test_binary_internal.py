import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import binary


def _build_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 7},
            "postprocess": {"calibration": "platt"},
        }
    )


def _build_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.1, 2.1, 0.2, 2.2, 0.3, 2.3, 0.4, 2.4],
            "x2": [1.0, 0.1, 1.1, 0.0, 0.9, 0.2, 1.2, 0.3],
            "group": [0, 0, 0, 1, 1, 1, 2, 2],
            "ts": [1, 2, 3, 4, 5, 6, 7, 8],
            "target": ["n", "y", "n", "y", "n", "y", "n", "y"],
        }
    )


def test_binary_build_feature_frame_validation_errors() -> None:
    config = _build_config()
    frame = _build_frame()

    with pytest.raises(VeldraValidationError):
        binary._build_feature_frame(config, frame.drop(columns=["target"]))
    with pytest.raises(VeldraValidationError):
        binary._build_feature_frame(config, frame.iloc[0:0])

    bad = frame.copy()
    bad.loc[0, "target"] = np.nan
    with pytest.raises(VeldraValidationError):
        binary._build_feature_frame(config, bad)

    three_class = frame.copy()
    three_class.loc[0, "target"] = "m"
    with pytest.raises(VeldraValidationError):
        binary._build_feature_frame(config, three_class)

    config.data.drop_cols = ["x1", "x2", "group", "ts"]
    with pytest.raises(VeldraValidationError):
        binary._build_feature_frame(config, frame)


def test_binary_to_python_scalar_and_encoding_failure(monkeypatch) -> None:
    assert binary._to_python_scalar(np.int64(3)) == 3

    cfg = _build_config()
    frame = _build_frame()
    monkeypatch.setattr(binary, "_to_python_scalar", lambda value: "same")
    with pytest.raises(VeldraValidationError, match="Failed to encode"):
        binary._build_feature_frame(cfg, frame)


def test_train_binary_with_cv_early_validation() -> None:
    frame = _build_frame()

    non_binary_payload = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "dummy.csv", "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
    }
    non_binary = RunConfig.model_validate(non_binary_payload)
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(non_binary, frame)

    cfg = _build_config()
    cfg.data.path = None
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(cfg, frame)


def test_train_binary_with_cv_rejects_empty_split(monkeypatch, FakeBooster) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        binary,
        "iter_cv_splits",
        lambda config, data, x, y=None: [
            (np.array([], dtype=int), np.array([0, 1], dtype=int))
        ],
    )
    monkeypatch.setattr(binary, "_train_single_booster", lambda **kwargs: FakeBooster())
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(cfg, frame)


def test_train_binary_with_cv_rejects_nan_oof(monkeypatch, FakeBooster) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        binary,
        "iter_cv_splits",
        lambda config, data, x, y=None: [
            (np.array([0, 2, 3, 5], dtype=int), np.array([1, 4], dtype=int))
        ],
    )
    monkeypatch.setattr(
        binary,
        "_train_single_booster",
        lambda **kwargs: FakeBooster(pred_value=0.4),
    )
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(cfg, frame)


def test_train_binary_with_cv_timeseries_path(monkeypatch, FakeBooster) -> None:
    cfg = _build_config()
    cfg.split.type = "timeseries"  # type: ignore[assignment]
    cfg.split.time_col = "ts"
    frame = _build_frame()

    monkeypatch.setattr(
        binary,
        "iter_cv_splits",
        lambda config, data, x, y=None: [
            (np.array([2, 3, 4, 5, 6, 7], dtype=int), np.array([0, 1], dtype=int)),
            (np.array([0, 1, 4, 5, 6, 7], dtype=int), np.array([2, 3], dtype=int)),
            (np.array([0, 1, 2, 3, 6, 7], dtype=int), np.array([4, 5], dtype=int)),
            (np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([6, 7], dtype=int)),
        ],
    )
    monkeypatch.setattr(
        binary,
        "_train_single_booster",
        lambda **kwargs: FakeBooster(pred_value=0.4),
    )

    output = binary.train_binary_with_cv(cfg, frame)
    assert output.metrics["mean"]["auc"] >= 0.0
