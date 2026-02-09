import pandas as pd

from examples import prepare_demo_data_binary


class _DummyFetched:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {
                "mean_radius": [11.0, 12.0, 13.0],
                "mean_texture": [15.0, 14.0, 13.0],
                "target": [0, 1, 0],
            }
        )
        self.target = self.frame["target"]


def test_prepare_demo_data_binary_creates_csv_with_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        prepare_demo_data_binary,
        "load_breast_cancer",
        lambda **_: _DummyFetched(),
    )
    out_path = tmp_path / "breast_cancer_binary.csv"

    exit_code = prepare_demo_data_binary.main(["--out-path", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert "target" in loaded.columns
