from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import estimate_dr


def _panel_frame(n_units: int = 80, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 55, size=n_units)
    skill = rng.normal(size=n_units)
    p = 1.0 / (1.0 + np.exp(-(-0.8 + 0.04 * (age - 30) + 0.7 * skill)))
    treatment = rng.binomial(1, p, size=n_units)
    base = 10000 + 300 * (age - 30) + 1800 * skill + rng.normal(0, 1000, size=n_units)
    pre = base + rng.normal(0, 600, size=n_units)
    post = base + 900 + treatment * 1800 + rng.normal(0, 600, size=n_units)
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


def test_estimate_drdid_panel_smoke(tmp_path) -> None:
    frame = _panel_frame()
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
            },
            "export": {"artifact_dir": str(tmp_path)},
        }
    )

    assert result.method == "dr_did"
    assert result.estimand == "att"
    assert result.metadata["design"] == "panel"
    assert {"naive", "ipw", "dr", "drdid"} <= set(result.metrics)
    assert {"overlap_metric", "smd_max_unweighted", "smd_max_weighted"} <= set(result.metrics)
    assert result.metadata["n_pre"] > 0
    assert result.metadata["n_post"] > 0
