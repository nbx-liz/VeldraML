import pandas as pd

from examples import prepare_demo_data_frontier


def test_prepare_demo_data_frontier_writes_csv(tmp_path) -> None:
    out_path = tmp_path / "frontier_demo.csv"
    exit_code = prepare_demo_data_frontier.main(["--out-path", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()

    frame = pd.read_csv(out_path)
    assert "target" in frame.columns
    assert len(frame) > 0
