import numpy as np
import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import binary


class _FakeBooster:
    def __init__(self, fold_value: float) -> None:
        self.best_iteration = 1
        self._fold_value = fold_value

    def predict(self, x: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        _ = num_iteration
        return np.full(len(x), self._fold_value, dtype=float)

    def model_to_string(self) -> str:
        return "fake-binary-model"


class _SpyCalibrator:
    def __init__(self) -> None:
        self.fit_x: np.ndarray | None = None
        self.fit_y: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_SpyCalibrator":
        self.fit_x = np.asarray(x, dtype=float).reshape(-1)
        self.fit_y = np.asarray(y, dtype=int).reshape(-1)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = np.clip(np.asarray(x, dtype=float).reshape(-1), 1e-7, 1 - 1e-7)
        return np.column_stack([1.0 - raw, raw])


def test_binary_calibrator_is_fit_on_oof_predictions(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.5, 0.4, 0.3, 0.2],
            "target": [0, 1, 0, 1],
        }
    )
    config = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 1},
            "postprocess": {"calibration": "platt"},
        }
    )

    fixed_splits = [
        (np.array([0, 1], dtype=int), np.array([2, 3], dtype=int)),
        (np.array([2, 3], dtype=int), np.array([0, 1], dtype=int)),
    ]
    monkeypatch.setattr(binary, "_iter_cv_splits", lambda cfg, data, x, y: fixed_splits)

    fold_values = iter([0.2, 0.8, 0.5])
    monkeypatch.setattr(
        binary,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(next(fold_values)),
    )

    spy = _SpyCalibrator()
    monkeypatch.setattr(binary, "LogisticRegression", lambda **kwargs: spy)

    output = binary.train_binary_with_cv(config, frame)

    assert spy.fit_x is not None
    assert spy.fit_y is not None
    expected_oof = np.array([0.8, 0.8, 0.2, 0.2], dtype=float)
    assert np.allclose(spy.fit_x, expected_oof)
    assert np.array_equal(spy.fit_y, frame["target"].to_numpy())
    assert output.calibrator is spy
