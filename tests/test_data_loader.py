import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.data.loader import load_tabular_data


def test_load_tabular_data_supports_csv_and_parquet(tmp_path) -> None:
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    csv_path = tmp_path / "data.csv"
    parquet_path = tmp_path / "data.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)

    loaded_csv = load_tabular_data(str(csv_path))
    loaded_parquet = load_tabular_data(str(parquet_path))

    assert loaded_csv.equals(frame)
    assert loaded_parquet.equals(frame)


def test_load_tabular_data_rejects_unsupported_extension(tmp_path) -> None:
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("x,y\n1,2\n", encoding="utf-8")

    with pytest.raises(VeldraValidationError):
        load_tabular_data(str(txt_path))
