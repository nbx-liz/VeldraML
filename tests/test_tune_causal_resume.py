from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import tune


def _dr_frame(rows: int = 72, seed: int = 555) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    p = 1.0 / (1.0 + np.exp(-(0.5 * x1 - 0.3 * x2)))
    treatment = rng.binomial(1, p)
    outcome = 1.0 + 0.8 * x1 - 0.6 * x2 + 1.2 * treatment + rng.normal(0, 0.4, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "outcome": outcome})


def test_causal_tune_resume_continues_trials(tmp_path) -> None:
    frame = _dr_frame()
    path = tmp_path / "dr.csv"
    frame.to_csv(path, index=False)
    study_name = "causal_resume_case"
    artifacts_dir = tmp_path / "artifacts"

    first = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 3},
            "causal": {"method": "dr", "treatment_col": "treatment"},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "objective": "dr_std_error",
                "study_name": study_name,
                "resume": False,
            },
            "export": {"artifact_dir": str(artifacts_dir)},
        }
    )
    first_trials = pd.read_parquet(Path(first.metadata["trials_path"]))

    second = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 3},
            "causal": {"method": "dr", "treatment_col": "treatment"},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "objective": "dr_std_error",
                "study_name": study_name,
                "resume": True,
            },
            "export": {"artifact_dir": str(artifacts_dir)},
        }
    )
    second_trials = pd.read_parquet(Path(second.metadata["trials_path"]))
    assert len(second_trials) >= len(first_trials) + 1
    assert second.metadata["resume"] is True
