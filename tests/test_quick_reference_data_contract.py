from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def _assert_dataset(
    filename: str,
    required_cols: set[str],
    min_rows: int,
    target_col: str,
    target_kind: str,
) -> pd.DataFrame:
    path = DATA_DIR / filename
    assert path.exists(), path
    frame = pd.read_csv(path)
    assert len(frame) >= min_rows, filename
    assert required_cols.issubset(set(frame.columns)), filename
    if target_kind == "numeric":
        assert pd.api.types.is_numeric_dtype(frame[target_col]), filename
    elif target_kind == "categorical":
        assert not pd.api.types.is_numeric_dtype(frame[target_col]), filename
    else:
        raise AssertionError(f"Unsupported target_kind: {target_kind}")
    return frame


def test_quick_reference_core_datasets_exist_with_contracts() -> None:
    ames = _assert_dataset(
        "ames_housing.csv",
        {"LotArea", "Neighborhood", "HouseStyle", "BldgType", "SalePrice"},
        min_rows=1000,
        target_col="SalePrice",
        target_kind="numeric",
    )
    titanic = _assert_dataset(
        "titanic.csv",
        {"Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"},
        min_rows=500,
        target_col="Survived",
        target_kind="numeric",
    )
    penguins = _assert_dataset(
        "penguins.csv",
        {"bill_length_mm", "flipper_length_mm", "island", "sex", "species", "body_mass_g"},
        min_rows=150,
        target_col="species",
        target_kind="categorical",
    )
    bike = _assert_dataset(
        "bike_sharing.csv",
        {"dteday", "season", "weathersit", "temp", "hum", "cnt"},
        min_rows=365,
        target_col="cnt",
        target_kind="numeric",
    )
    lalonde = _assert_dataset(
        "lalonde.csv",
        {"treatment", "outcome", "x1", "x2"},
        min_rows=100,
        target_col="outcome",
        target_kind="numeric",
    )
    cps_panel = _assert_dataset(
        "cps_panel.csv",
        {"unit_id", "time", "post", "treatment", "age", "skill", "target"},
        min_rows=100,
        target_col="target",
        target_kind="numeric",
    )

    assert set(titanic["Survived"].dropna().astype(int).unique().tolist()) <= {0, 1}
    assert ames["SalePrice"].min() > 0
    assert penguins["species"].nunique() >= 3
    pd.to_datetime(bike["dteday"])
    assert set(lalonde["treatment"].dropna().astype(int).unique().tolist()) <= {0, 1}
    assert set(cps_panel["treatment"].dropna().astype(int).unique().tolist()) <= {0, 1}
    assert set(cps_panel["post"].dropna().astype(int).unique().tolist()) <= {0, 1}
    assert cps_panel["time"].nunique() >= 2


def test_quick_reference_sources_manifest_has_required_fields() -> None:
    path = DATA_DIR / "quick_reference_sources.json"
    assert path.exists(), path
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", {})
    for key in ["lalonde", "cps_panel"]:
        assert key in datasets, key
        record = datasets[key]
        for field in [
            "source_url",
            "retrieved_at_utc",
            "sha256",
            "license_note",
            "transform_version",
        ]:
            assert record.get(field), f"{key}.{field}"
