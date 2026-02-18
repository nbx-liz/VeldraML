from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.data.loader import load_tabular_data


def test_load_tabular_data_propagates_csv_encoding_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_encoding.csv"
    path.write_bytes(b"x1,target\n1,\xff\n")
    with pytest.raises(UnicodeDecodeError):
        load_tabular_data(str(path))


def test_load_tabular_data_propagates_malformed_csv_error(tmp_path: Path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text('x1,target\n1,"unterminated\n2,3\n', encoding="utf-8")
    with pytest.raises(pd.errors.ParserError):
        load_tabular_data(str(path))


def test_load_tabular_data_propagates_invalid_parquet_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"this-is-not-a-valid-parquet")
    with pytest.raises(Exception):
        load_tabular_data(str(path))


def test_load_tabular_data_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tabular_data(str(tmp_path / "missing.csv"))


def test_load_tabular_data_propagates_permission_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "forbidden.csv"
    path.write_text("x1,target\n1,2\n", encoding="utf-8")

    def _raise_permission(*_args, **_kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(pd, "read_csv", _raise_permission)
    with pytest.raises(PermissionError):
        load_tabular_data(str(path))


def test_load_tabular_data_handles_large_csv_payload(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": list(range(20000)),
            "target": [v * 0.1 for v in range(20000)],
        }
    )
    path = tmp_path / "large.csv"
    frame.to_csv(path, index=False)
    loaded = load_tabular_data(str(path))
    assert len(loaded) == 20000
    assert loaded.columns.tolist() == ["x1", "target"]


def test_load_tabular_data_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(VeldraValidationError, match="Unsupported data format"):
        load_tabular_data(str(path))


def test_load_tabular_data_preserves_missing_values_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing_values.csv"
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, None],
            "x2": [10.0, None, 30.0],
            "target": [0.0, 1.0, 0.0],
        }
    )
    frame.to_csv(path, index=False)

    loaded = load_tabular_data(str(path))
    assert loaded.columns.tolist() == ["x1", "x2", "target"]
    assert loaded["x1"].isna().sum() == 1
    assert loaded["x2"].isna().sum() == 1
