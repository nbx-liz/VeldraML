import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import binary


def _config_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": "dummy.csv", "target": "target"},
        "split": {"type": "stratified", "n_splits": 3, "seed": 7},
        "postprocess": {"calibration": "platt"},
    }


def _build_config() -> RunConfig:
    return RunConfig.model_validate(_config_payload())


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


def test_binary_iter_cv_splits_validation_errors() -> None:
    config = _build_config()
    frame = _build_frame()
    x = frame[["x1", "x2"]]
    y = pd.Series([0, 0, 0, 1, 1, 1])

    config.split.type = "group"  # type: ignore[assignment]
    config.split.group_col = None
    with pytest.raises(VeldraValidationError):
        binary._iter_cv_splits(config, frame, x, y)

    config.split.group_col = "missing_group"
    with pytest.raises(VeldraValidationError):
        binary._iter_cv_splits(config, frame, x, y)

    config.split.type = "timeseries"  # type: ignore[assignment]
    config.split.time_col = None
    with pytest.raises(VeldraValidationError):
        binary._iter_cv_splits(config, frame, x, y)

    config.split.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraValidationError):
        binary._iter_cv_splits(config, frame, x, y)


def test_binary_iter_cv_splits_success_paths() -> None:
    config = _build_config()
    frame = _build_frame()
    x = frame[["x1", "x2"]]
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    config.split.type = "kfold"  # type: ignore[assignment]
    kfold_splits = binary._iter_cv_splits(config, frame, x, y)
    assert kfold_splits

    config.split.type = "group"  # type: ignore[assignment]
    config.split.group_col = "group"
    config.split.n_splits = 2
    group_splits = binary._iter_cv_splits(config, frame, x, y)
    assert group_splits

    config.split.type = "timeseries"  # type: ignore[assignment]
    config.split.time_col = "ts"
    ts_splits = binary._iter_cv_splits(config, frame, x, y)
    assert ts_splits


def test_train_binary_with_cv_early_validation() -> None:
    frame = _build_frame()

    non_binary_payload = _config_payload()
    non_binary_payload["task"] = {"type": "regression"}
    non_binary_payload.pop("postprocess", None)
    non_binary = RunConfig.model_validate(non_binary_payload)
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(non_binary, frame)

    cfg = _build_config()
    cfg.data.path = None
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(cfg, frame)


class _FakeBooster:
    def __init__(self, pred_value: float = 0.5) -> None:
        self.best_iteration = 1
        self._pred_value = pred_value

    def predict(self, x: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        _ = num_iteration
        return np.full(len(x), self._pred_value, dtype=float)

    def model_to_string(self) -> str:
        return "fake-model"


def test_train_binary_with_cv_rejects_empty_split(monkeypatch) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        binary,
        "_iter_cv_splits",
        lambda config, data, x, y: [(np.array([], dtype=int), np.array([0, 1], dtype=int))],
    )
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(cfg, frame)


def test_train_binary_with_cv_rejects_nan_oof(monkeypatch) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        binary,
        "_iter_cv_splits",
        lambda config, data, x, y: [
            (np.array([0, 2, 3, 5], dtype=int), np.array([1, 4], dtype=int))
        ],
    )
    monkeypatch.setattr(
        binary,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(pred_value=0.4),
    )
    with pytest.raises(VeldraValidationError):
        binary.train_binary_with_cv(cfg, frame)


def test_train_binary_with_cv_timeseries_path(monkeypatch) -> None:
    cfg = _build_config()
    cfg.split.type = "timeseries"  # type: ignore[assignment]
    cfg.split.time_col = "ts"
    frame = _build_frame()

    monkeypatch.setattr(
        binary,
        "_iter_cv_splits",
        lambda config, data, x, y: [
            (np.array([2, 3, 4, 5, 6, 7], dtype=int), np.array([0, 1], dtype=int)),
            (np.array([0, 1, 4, 5, 6, 7], dtype=int), np.array([2, 3], dtype=int)),
            (np.array([0, 1, 2, 3, 6, 7], dtype=int), np.array([4, 5], dtype=int)),
            (np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([6, 7], dtype=int)),
        ],
    )
    monkeypatch.setattr(
        binary,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(pred_value=0.4),
    )

    output = binary.train_binary_with_cv(cfg, frame)
    assert output.metrics["mean"]["auc"] >= 0.0


def test_binary_timeseries_splitter_receives_extended_params(monkeypatch) -> None:
    config = _build_config()
    config.split.type = "timeseries"  # type: ignore[assignment]
    config.split.time_col = "ts"
    config.split.timeseries_mode = "blocked"
    config.split.test_size = 2
    config.split.gap = 1
    config.split.embargo = 2
    config.split.train_size = 3

    frame = _build_frame()
    x = frame[["x1", "x2"]]
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
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

        def split(self, data: int) -> list[tuple[np.ndarray, np.ndarray]]:
            _ = data
            return [(np.array([0, 1, 2], dtype=int), np.array([3, 4], dtype=int))]

    monkeypatch.setattr(binary, "TimeSeriesSplitter", _FakeSplitter)
    splits = binary._iter_cv_splits(config, frame, x, y)
    assert splits
    assert captured == {
        "n_splits": config.split.n_splits,
        "test_size": 2,
        "gap": 1,
        "embargo": 2,
        "mode": "blocked",
        "train_size": 3,
    }
