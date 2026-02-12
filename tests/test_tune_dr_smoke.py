from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import tune


def _dr_frame(rows: int = 80, seed: int = 333) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    p = 1.0 / (1.0 + np.exp(-(0.4 * x1 - 0.3 * x2)))
    treatment = rng.binomial(1, p)
    outcome = 2.0 + 1.1 * x1 - 0.7 * x2 + 1.5 * treatment + rng.normal(scale=0.5, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome})


def test_tune_supports_dr_objective(tmp_path) -> None:
    frame = _dr_frame()
    path = tmp_path / "dr.csv"
    frame.to_csv(path, index=False)

    result = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "causal": {"method": "dr", "treatment_col": "treatment"},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "objective": "dr_balance_priority",
                "causal_penalty_weight": 0.5,
                "causal_balance_threshold": 0.10,
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert result.metadata["metric_name"] == "dr_balance_priority"
    trials_path = Path(result.metadata["trials_path"])
    trials = pd.read_parquet(trials_path)
    assert {
        "estimate",
        "std_error",
        "overlap_metric",
        "smd_max_unweighted",
        "smd_max_weighted",
        "balance_threshold",
        "balance_violation",
        "penalty",
        "objective_value",
        "objective_stage",
    } <= set(trials.columns)

