import json

import numpy as np
import pandas as pd
import pytest

from examples import run_demo_frontier


def _frontier_frame(rows: int = 140, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.2, 2.2, size=rows)
    x2 = rng.normal(size=rows)
    y = (
        1.2
        + 1.4 * x1
        - 0.3 * x2
        + rng.normal(scale=0.2, size=rows)
        + rng.exponential(scale=0.3, size=rows)
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_run_demo_frontier_writes_expected_outputs(tmp_path) -> None:
    data_path = tmp_path / "demo_frontier.csv"
    out_dir = tmp_path / "out"
    _frontier_frame().to_csv(data_path, index=False)

    exit_code = run_demo_frontier.main(
        [
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "11",
            "--n-splits",
            "3",
            "--alpha",
            "0.9",
        ]
    )

    assert exit_code == 0
    run_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "run_result.json").exists()
    assert (run_dir / "eval_result.json").exists()
    assert (run_dir / "predictions_sample.csv").exists()
    assert (run_dir / "used_config.yaml").exists()

    run_result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    assert run_result["artifact_path"]

    pred_sample = pd.read_csv(run_dir / "predictions_sample.csv")
    assert "frontier_pred" in pred_sample.columns


def test_run_demo_frontier_requires_prepared_csv(tmp_path) -> None:
    missing_path = tmp_path / "missing_frontier.csv"
    with pytest.raises(SystemExit) as exc_info:
        run_demo_frontier.main(["--data-path", str(missing_path)])

    assert "prepare_demo_data_frontier.py" in str(exc_info.value)
