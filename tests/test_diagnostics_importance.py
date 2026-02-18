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


class _WrappedBooster:
    def __init__(self, inner) -> None:
        self.booster_ = inner


class _NamesOnlyBooster:
    def feature_name(self):
        return ["a", "b"]


class _MismatchBooster:
    def feature_name(self):
        return ["a", "b", "c"]

    def feature_importance(self, importance_type: str = "gain"):
        _ = importance_type
        return np.array([1.0])


def test_compute_importance_resolves_booster_attribute() -> None:
    out = compute_importance(_WrappedBooster(_FakeBooster()))
    assert list(out["feature"]) == ["b", "c", "a"]


def test_compute_importance_returns_empty_frame_for_invalid_booster() -> None:
    out = compute_importance(_NamesOnlyBooster())
    assert list(out.columns) == ["feature", "importance"]
    assert out.empty


def test_compute_importance_returns_empty_frame_for_mismatched_lengths() -> None:
    out = compute_importance(_MismatchBooster())
    assert list(out.columns) == ["feature", "importance"]
    assert out.empty
