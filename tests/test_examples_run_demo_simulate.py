from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from examples import run_demo_simulate


def _regression_frame(rows: int = 100, seed: int = 601) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 2.2 * x1 - 1.0 * x2 + rng.normal(scale=0.25, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_run_demo_simulate_writes_expected_outputs(tmp_path) -> None:
    data_path = tmp_path / "simulate_demo.csv"
    out_dir = tmp_path / "out"
    _regression_frame().to_csv(data_path, index=False)

    exit_code = run_demo_simulate.main(
        [
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "11",
            "--n-splits",
            "3",
        ]
    )

    assert exit_code == 0
    run_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "simulate_result.csv").exists()
    assert (run_dir / "simulate_summary.json").exists()
    assert (run_dir / "used_scenarios.yaml").exists()

    summary = json.loads((run_dir / "simulate_summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"]
    assert summary["artifact_path"]

    result = pd.read_csv(run_dir / "simulate_result.csv")
    assert {"row_id", "scenario", "task_type", "base_pred", "scenario_pred", "delta_pred"} <= set(
        result.columns
    )


def test_run_demo_simulate_requires_prepared_csv(tmp_path) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(SystemExit) as exc_info:
        run_demo_simulate.main(["--data-path", str(missing_path)])
    assert "prepare_demo_data.py" in str(exc_info.value)
