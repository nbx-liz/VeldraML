import pandas as pd

from examples import prepare_demo_data_multiclass


class _DummyFetched:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {
                "sepal length (cm)": [5.1, 5.9, 6.7],
                "sepal width (cm)": [3.5, 3.0, 3.1],
                "target": [0, 1, 2],
            }
        )
        self.target_names = ["setosa", "versicolor", "virginica"]


def test_prepare_demo_data_multiclass_creates_csv_with_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prepare_demo_data_multiclass,
        "load_iris",
        lambda **_: _DummyFetched(),
    )
    out_path = tmp_path / "iris_multiclass.csv"

    exit_code = prepare_demo_data_multiclass.main(["--out-path", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert "target" in loaded.columns
    assert set(loaded["target"]) == {"setosa", "versicolor", "virginica"}
