from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import Artifact, fit, simulate


def _binary_frame(rows: int = 80, seed: int = 411) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 0.9 * x1 - 0.7 * x2 + rng.normal(scale=0.35, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_simulate_binary_predict_contract(tmp_path) -> None:
    frame = _binary_frame()
    data_path = tmp_path / "binary_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 4, "seed": 42},
            "postprocess": {"calibration": "platt"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    artifact = Artifact.load(run.artifact_path)
    scenarios = [{"name": "x1_shift", "actions": [{"op": "add", "column": "x1", "value": 0.4}]}]
    sim_result = simulate(artifact, frame, scenarios)

    out = sim_result.data
    assert {"base_p_cal", "scenario_p_cal", "delta_p_cal"} <= set(out.columns)
    assert {"base_label_pred", "scenario_label_pred", "label_changed"} <= set(out.columns)
    assert out["base_p_cal"].between(0.0, 1.0).all()
    assert out["scenario_p_cal"].between(0.0, 1.0).all()
