import json

import numpy as np
import pandas as pd
import pytest

from examples import evaluate_demo_frontier_artifact, run_demo_frontier


def _frontier_frame(rows: int = 120, seed: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    y = (
        1.4
        + 1.2 * x1
        - 0.6 * x2
        + rng.normal(scale=0.25, size=rows)
        + rng.exponential(scale=0.2, size=rows)
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_evaluate_demo_frontier_artifact_writes_eval_result(tmp_path) -> None:
    data_path = tmp_path / "frontier.csv"
    out_dir = tmp_path / "run_out"
    _frontier_frame().to_csv(data_path, index=False)

    run_demo_frontier.main(
        ["--data-path", str(data_path), "--out-dir", str(out_dir), "--n-splits", "3"]
    )
    run_dir = sorted([p for p in out_dir.iterdir() if p.is_dir()])[0]
    run_result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    artifact_path = run_result["artifact_path"]

    eval_out_dir = tmp_path / "eval_out"
    exit_code = evaluate_demo_frontier_artifact.main(
        [
            "--artifact-path",
            artifact_path,
            "--data-path",
            str(data_path),
            "--out-dir",
            str(eval_out_dir),
        ]
    )

    assert exit_code == 0
    eval_dirs = [path for path in eval_out_dir.iterdir() if path.is_dir()]
    assert len(eval_dirs) == 1
    eval_result = json.loads((eval_dirs[0] / "eval_only_result.json").read_text(encoding="utf-8"))
    assert {"pinball", "mae", "mean_u_hat", "coverage"} <= set(eval_result["metrics"])


def test_evaluate_demo_frontier_artifact_fails_on_missing_target(tmp_path) -> None:
    data_path = tmp_path / "frontier.csv"
    out_dir = tmp_path / "run_out"
    frame = _frontier_frame()
    frame.to_csv(data_path, index=False)

    run_demo_frontier.main(
        ["--data-path", str(data_path), "--out-dir", str(out_dir), "--n-splits", "3"]
    )
    run_dir = sorted([p for p in out_dir.iterdir() if p.is_dir()])[0]
    run_result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    artifact_path = run_result["artifact_path"]

    invalid_path = tmp_path / "invalid.csv"
    frame.drop(columns=["target"]).to_csv(invalid_path, index=False)

    with pytest.raises(SystemExit):
        evaluate_demo_frontier_artifact.main(
            ["--artifact-path", artifact_path, "--data-path", str(invalid_path)]
        )
