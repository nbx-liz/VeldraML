import pandas as pd

from examples import prepare_demo_data


class _DummyFetched:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {
                "MedInc": [1.1, 1.2, 1.3],
                "HouseAge": [10, 20, 30],
                "MedHouseVal": [2.5, 2.7, 2.9],
            }
        )
        self.target = self.frame["MedHouseVal"]


def test_prepare_demo_data_creates_csv_with_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prepare_demo_data,
        "fetch_california_housing",
        lambda **_: _DummyFetched(),
    )
    out_path = tmp_path / "california.csv"

    exit_code = prepare_demo_data.main(["--out-path", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert "target" in loaded.columns
