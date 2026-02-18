from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.diagnostics.shap_native import compute_shap, compute_shap_multiclass


class _FakeBooster:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def predict(self, x: pd.DataFrame, pred_contrib: bool = False):
        _ = x
        _ = pred_contrib
        return self._matrix


def test_compute_shap_returns_feature_aligned_frame() -> None:
    x = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    contrib = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    out = compute_shap(_FakeBooster(contrib), x)
    assert list(out.columns) == ["f1", "f2"]
    assert out.shape == (2, 2)


def test_compute_shap_multiclass_selects_predicted_class() -> None:
    x = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    n_classes = 3
    n_features = 2
    raw = np.zeros((1, (n_features + 1) * n_classes), dtype=float)
    raw[0, 3:5] = [0.7, 0.9]  # class 1 contribs for f1,f2

    out = compute_shap_multiclass(
        _FakeBooster(raw),
        x,
        predictions=np.array([1]),
        n_classes=3,
    )
    assert list(out.columns) == ["f1", "f2"]
    assert float(out.iloc[0, 0]) == 0.7
    assert float(out.iloc[0, 1]) == 0.9
class _WrappedPredictBooster:
    def __init__(self, matrix: np.ndarray) -> None:
        self.booster_ = _FakeBooster(matrix)


def test_compute_shap_resolves_booster_attribute() -> None:
    x = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    raw = np.array([[0.3, 0.4, 0.5]])
    out = compute_shap(_WrappedPredictBooster(raw), x)
    assert out.shape == (1, 2)


def test_compute_shap_raises_on_non_2d_contrib() -> None:
    x = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    with pytest.raises(ValueError, match="Expected 2D"):
        compute_shap(_FakeBooster(np.array([0.1, 0.2, 0.3])), x)


def test_compute_shap_raises_on_invalid_contrib_width() -> None:
    x = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    with pytest.raises(ValueError, match="Unexpected pred_contrib width"):
        compute_shap(_FakeBooster(np.array([[0.1, 0.2]])), x)


def test_compute_shap_multiclass_rejects_shape_and_length_mismatch() -> None:
    x = pd.DataFrame({"f1": [1.0, 2.0], "f2": [2.0, 3.0]})

    wrong_shape = np.array([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="Unexpected multiclass pred_contrib shape"):
        compute_shap_multiclass(
            _FakeBooster(wrong_shape),
            x,
            predictions=np.array([0, 1]),
            n_classes=2,
        )

    ok_shape = np.zeros((2, (2 + 1) * 2), dtype=float)
    with pytest.raises(ValueError, match="predictions length must match X rows"):
        compute_shap_multiclass(_FakeBooster(ok_shape), x, predictions=np.array([0]), n_classes=2)
