from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import Artifact, fit, simulate


def _frontier_frame(rows: int = 90, seed: int = 413) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.8, 1.8, size=rows)
    x2 = rng.normal(size=rows)
    y = 1.1 + 1.7 * x1 - 0.5 * x2 + rng.normal(scale=0.18, size=rows) + rng.exponential(
        scale=0.25, size=rows
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_simulate_frontier_u_hat_is_conditional(tmp_path) -> None:
    frame = _frontier_frame()
    data_path = tmp_path / "frontier_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 42},
            "frontier": {"alpha": 0.90},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    artifact = Artifact.load(run.artifact_path)
    scenario = {"name": "x1_up", "actions": [{"op": "add", "column": "x1", "value": 0.3}]}

    with_target = simulate(artifact, frame, [scenario]).data
    assert {"base_pred", "scenario_pred", "delta_pred"} <= set(with_target.columns)
    assert {"base_u_hat", "scenario_u_hat", "delta_u_hat"} <= set(with_target.columns)

    without_target = simulate(artifact, frame.drop(columns=["target"]), [scenario]).data
    assert {"base_pred", "scenario_pred", "delta_pred"} <= set(without_target.columns)
    assert "base_u_hat" not in without_target.columns
