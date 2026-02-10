import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import multiclass


def _config_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "multiclass"},
        "data": {"path": "dummy.csv", "target": "target"},
        "split": {"type": "stratified", "n_splits": 3, "seed": 7},
    }


def _build_config() -> RunConfig:
    return RunConfig.model_validate(_config_payload())


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


def test_iter_cv_splits_validation_errors() -> None:
    config = _build_config()
    frame = _build_frame()
    x = frame[["x1", "x2"]]
    y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2])

    config.split.type = "group"  # type: ignore[assignment]
    config.split.group_col = None
    with pytest.raises(VeldraValidationError):
        multiclass._iter_cv_splits(config, frame, x, y)

    config.split.group_col = "missing_group"
    with pytest.raises(VeldraValidationError):
        multiclass._iter_cv_splits(config, frame, x, y)

    config.split.type = "timeseries"  # type: ignore[assignment]
    config.split.time_col = None
    with pytest.raises(VeldraValidationError):
        multiclass._iter_cv_splits(config, frame, x, y)

    config.split.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraValidationError):
        multiclass._iter_cv_splits(config, frame, x, y)


def test_iter_cv_splits_success_paths() -> None:
    config = _build_config()
    frame = _build_frame()
    x = frame[["x1", "x2"]]
    y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2])

    config.split.type = "kfold"  # type: ignore[assignment]
    kfold_splits = multiclass._iter_cv_splits(config, frame, x, y)
    assert kfold_splits

    config.split.type = "group"  # type: ignore[assignment]
    config.split.group_col = "group"
    config.split.n_splits = 3
    group_splits = multiclass._iter_cv_splits(config, frame, x, y)
    assert group_splits

    config.split.type = "timeseries"  # type: ignore[assignment]
    config.split.time_col = "ts"
    ts_splits = multiclass._iter_cv_splits(config, frame, x, y)
    assert ts_splits


def test_normalize_proba_validation_errors() -> None:
    with pytest.raises(VeldraValidationError):
        multiclass._normalize_proba(np.array([0.1, 0.2, 0.3]), n_rows=2, num_class=3)
    with pytest.raises(VeldraValidationError):
        multiclass._normalize_proba(np.ones((4, 2)), n_rows=4, num_class=3)


def test_train_multiclass_with_cv_early_validation() -> None:
    frame = _build_frame()

    non_mc_payload = _config_payload()
    non_mc_payload["task"] = {"type": "binary"}
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
        "_iter_cv_splits",
        lambda config, data, x, y: [(np.array([], dtype=int), np.array([0, 1], dtype=int))],
    )

    with pytest.raises(VeldraValidationError):
        multiclass.train_multiclass_with_cv(cfg, frame)


def test_train_multiclass_with_cv_rejects_nan_oof(monkeypatch) -> None:
    cfg = _build_config()
    frame = _build_frame()

    monkeypatch.setattr(
        multiclass,
        "_iter_cv_splits",
        lambda config, data, x, y: [
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
        "_iter_cv_splits",
        lambda config, data, x, y: [
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


def test_multiclass_timeseries_splitter_receives_extended_params(monkeypatch) -> None:
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
    y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2])
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

    monkeypatch.setattr(multiclass, "TimeSeriesSplitter", _FakeSplitter)
    splits = multiclass._iter_cv_splits(config, frame, x, y)
    assert splits
    assert captured == {
        "n_splits": config.split.n_splits,
        "test_size": 2,
        "gap": 1,
        "embargo": 2,
        "mode": "blocked",
        "train_size": 3,
    }
