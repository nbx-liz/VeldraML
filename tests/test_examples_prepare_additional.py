from __future__ import annotations

import pandas as pd
import pytest

from examples import (
    prepare_demo_data,
    prepare_demo_data_binary,
    prepare_demo_data_multiclass,
)


def test_build_california_frame_uses_target_attribute_when_column_missing(monkeypatch) -> None:
    class _Fetched:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"f1": [1.0, 2.0]})
            self.target = pd.Series([10.0, 20.0])

    monkeypatch.setattr(prepare_demo_data, "fetch_california_housing", lambda **_: _Fetched())
    frame = prepare_demo_data.build_california_frame()
    assert "target" in frame.columns
    assert frame["target"].tolist() == [10.0, 20.0]


def test_build_california_frame_raises_when_target_is_unavailable(monkeypatch) -> None:
    class _Fetched:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"f1": [1.0, 2.0]})

    monkeypatch.setattr(prepare_demo_data, "fetch_california_housing", lambda **_: _Fetched())
    with pytest.raises(ValueError, match="Could not determine target column"):
        prepare_demo_data.build_california_frame()


def test_build_breast_cancer_frame_adds_target_from_attribute(monkeypatch) -> None:
    class _Fetched:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"f1": [1.0, 2.0]})
            self.target = pd.Series([0, 1])

    monkeypatch.setattr(prepare_demo_data_binary, "load_breast_cancer", lambda **_: _Fetched())
    frame = prepare_demo_data_binary.build_breast_cancer_frame()
    assert "target" in frame.columns
    assert frame["target"].tolist() == [0, 1]


def test_build_breast_cancer_frame_raises_when_target_is_unavailable(monkeypatch) -> None:
    class _Fetched:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"f1": [1.0, 2.0]})

    monkeypatch.setattr(prepare_demo_data_binary, "load_breast_cancer", lambda **_: _Fetched())
    with pytest.raises(ValueError, match="Could not determine target column"):
        prepare_demo_data_binary.build_breast_cancer_frame()


def test_build_iris_frame_renames_target_when_default_target_changes(monkeypatch) -> None:
    class _Fetched:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 1]})
            self.target_names = ["a", "b", "c"]

    monkeypatch.setattr(prepare_demo_data_multiclass, "DEFAULT_TARGET", "species")
    monkeypatch.setattr(prepare_demo_data_multiclass, "load_iris", lambda **_: _Fetched())
    frame = prepare_demo_data_multiclass.build_iris_frame()
    assert "species" in frame.columns
    assert set(frame["species"]) == {"a", "b"}


def test_build_iris_frame_raises_when_mapping_produces_null(monkeypatch) -> None:
    class _Fetched:
        def __init__(self) -> None:
            self.frame = pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 9]})
            self.target_names = ["a", "b", "c"]

    monkeypatch.setattr(prepare_demo_data_multiclass, "load_iris", lambda **_: _Fetched())
    with pytest.raises(ValueError, match="Target mapping produced null values"):
        prepare_demo_data_multiclass.build_iris_frame()
