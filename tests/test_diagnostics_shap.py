from __future__ import annotations

import numpy as np
import pandas as pd

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
