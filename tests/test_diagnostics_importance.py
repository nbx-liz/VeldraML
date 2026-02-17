from __future__ import annotations

import numpy as np

from veldra.diagnostics.importance import compute_importance


class _FakeBooster:
    def feature_name(self):
        return ["a", "b", "c"]

    def feature_importance(self, importance_type: str = "gain"):
        _ = importance_type
        return np.array([1.0, 3.0, 2.0])


def test_compute_importance_returns_sorted_dataframe() -> None:
    out = compute_importance(_FakeBooster(), importance_type="gain", top_n=2)
    assert list(out.columns) == ["feature", "importance"]
    assert list(out["feature"]) == ["b", "c"]
