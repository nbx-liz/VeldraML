from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import Artifact, fit, simulate


def _regression_frame(rows: int = 60, seed: int = 410) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.2 * x1 - 0.8 * x2 + rng.normal(scale=0.2, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_simulate_regression_returns_long_form_result(tmp_path) -> None:
    frame = _regression_frame()
    data_path = tmp_path / "reg_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 42},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    artifact = Artifact.load(run.artifact_path)
    scenarios = [
        {"name": "x1_plus", "actions": [{"op": "add", "column": "x1", "value": 0.3}]},
        {"name": "x2_scale", "actions": [{"op": "mul", "column": "x2", "value": 1.2}]},
    ]

    sim_result = simulate(artifact, frame, scenarios)
    out = sim_result.data

    assert sim_result.task_type == "regression"
    assert sim_result.metadata["n_rows"] == len(frame)
    assert sim_result.metadata["n_scenarios"] == 2
    assert len(out) == len(frame) * 2
    assert {"row_id", "scenario", "task_type", "base_pred", "scenario_pred", "delta_pred"} <= set(
        out.columns
    )
