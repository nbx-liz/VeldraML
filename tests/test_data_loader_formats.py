from pathlib import Path

import pandas as pd
import pytest

from veldra.api import VeldraValidationError
from veldra.data import load_tabular_data


def test_load_tabular_data_supports_csv_and_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    csv_path = tmp_path / "sample.csv"
    parquet_path = tmp_path / "sample.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    loaded_csv = load_tabular_data(str(csv_path))
    loaded_parquet = load_tabular_data(str(parquet_path))

    assert loaded_csv.equals(frame)
    assert loaded_parquet.equals(frame)


def test_load_tabular_data_rejects_unsupported_extension(tmp_path: Path) -> None:
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(VeldraValidationError):
        load_tabular_data(str(txt_path))
