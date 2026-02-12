from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune


def _dr_frame(rows: int = 70, seed: int = 81) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    treatment = rng.binomial(1, p)
    outcome = 1.0 + 1.0 * x1 - 0.7 * x2 + 1.3 * treatment + rng.normal(0, 0.5, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome})


def _drdid_panel_frame(n_units: int = 64, seed: int = 82) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 58, size=n_units)
    skill = rng.normal(size=n_units)
    p_t = 1.0 / (1.0 + np.exp(-(-0.7 + 0.03 * (age - 30) + 0.6 * skill)))
    treatment = rng.binomial(1, p_t, size=n_units)
    base = 9000 + 210 * (age - 30) + 1300 * skill + rng.normal(0, 800, size=n_units)
    pre = base + rng.normal(0, 500, size=n_units)
    post = base + 650 + 1200 * treatment + rng.normal(0, 500, size=n_units)
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


def test_causal_tune_defaults_to_balance_priority_objectives(tmp_path) -> None:
    dr_path = tmp_path / "dr.csv"
    _dr_frame().to_csv(dr_path, index=False)
    dr_result = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(dr_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 3},
            "causal": {"method": "dr", "treatment_col": "treatment"},
            "tuning": {"enabled": True, "n_trials": 1},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert dr_result.metadata["metric_name"] == "dr_balance_priority"

    drdid_path = tmp_path / "drdid.csv"
    _drdid_panel_frame().to_csv(drdid_path, index=False)
    drdid_result = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(drdid_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 3},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
            },
            "tuning": {"enabled": True, "n_trials": 1},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert drdid_result.metadata["metric_name"] == "drdid_balance_priority"
