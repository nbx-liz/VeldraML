from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import tune


def _panel_frame(n_units: int = 60, seed: int = 444) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 55, size=n_units)
    skill = rng.normal(size=n_units)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 0.03 * (age - 30) + 0.7 * skill)))
    treatment = rng.binomial(1, p, size=n_units)
    base = 9000 + 250 * (age - 30) + 1300 * skill + rng.normal(0, 800, size=n_units)
    pre = base + rng.normal(0, 500, size=n_units)
    post = base + 700 + treatment * 1400 + rng.normal(0, 500, size=n_units)
    pre_df = pd.DataFrame(
        {
            "unit_id": np.arange(n_units),
            "time": 0,
            "post": 0,
            "treatment": treatment,
            "age": age,
            "skill": skill,
            "outcome": pre,
        }
    )
    post_df = pre_df.copy()
    post_df["time"] = 1
    post_df["post"] = 1
    post_df["outcome"] = post
    return pd.concat([pre_df, post_df], ignore_index=True)


def test_tune_supports_drdid_objective(tmp_path) -> None:
    frame = _panel_frame()
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 2},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
            },
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "objective": "drdid_balance_priority",
                "causal_balance_threshold": 0.10,
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert result.metadata["metric_name"] == "drdid_balance_priority"
    trials = pd.read_parquet(Path(result.metadata["trials_path"]))
    assert {
        "estimate",
        "std_error",
        "smd_max_unweighted",
        "smd_max_weighted",
        "balance_threshold",
        "balance_violation",
        "objective_value",
        "objective_stage",
    } <= set(trials.columns)

