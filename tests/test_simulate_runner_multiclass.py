from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import Artifact, fit, simulate


def _multiclass_frame(rows_per_class: int = 20, seed: int = 412) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["alpha", "beta", "gamma"]
    parts: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        x1 = rng.normal(loc=idx * 1.4, scale=0.35, size=rows_per_class)
        x2 = rng.normal(loc=1.5 - idx * 1.2, scale=0.35, size=rows_per_class)
        parts.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(parts, ignore_index=True)


def test_simulate_multiclass_predict_contract(tmp_path) -> None:
    frame = _multiclass_frame()
    data_path = tmp_path / "multiclass_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 42},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    artifact = Artifact.load(run.artifact_path)
    scenarios = [{"name": "x2_scale", "actions": [{"op": "mul", "column": "x2", "value": 1.1}]}]
    sim_result = simulate(artifact, frame, scenarios)
    out = sim_result.data

    assert {"base_label_pred", "scenario_label_pred", "label_changed"} <= set(out.columns)
    proba_cols = [col for col in out.columns if col.startswith("base_proba_")]
    assert proba_cols
    for base_col in proba_cols:
        suffix = base_col.replace("base_", "")
        assert f"scenario_{suffix}" in out.columns
        assert f"delta_{suffix}" in out.columns
