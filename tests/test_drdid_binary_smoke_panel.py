from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import estimate_dr


def _panel_binary_frame(n_units: int = 120, seed: int = 731) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 60, size=n_units)
    skill = rng.normal(size=n_units)
    logits_t = -0.9 + 0.05 * (age - 30) + 0.8 * skill
    p_t = 1.0 / (1.0 + np.exp(-logits_t))
    treatment = rng.binomial(1, p_t, size=n_units)

    logits_pre = -1.3 + 0.03 * (age - 30) + 0.5 * skill
    logits_post = logits_pre + 0.2 + 0.7 * treatment
    y_pre = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits_pre)), size=n_units)
    y_post = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits_post)), size=n_units)

    pre_df = pd.DataFrame(
        {
            "unit_id": np.arange(n_units),
            "time": 0,
            "post": 0,
            "treatment": treatment,
            "age": age,
            "skill": skill,
            "outcome": y_pre,
        }
    )
    post_df = pre_df.copy()
    post_df["time"] = 1
    post_df["post"] = 1
    post_df["outcome"] = y_post
    return pd.concat([pre_df, post_df], ignore_index=True)


def test_estimate_drdid_binary_panel_smoke(tmp_path) -> None:
    frame = _panel_binary_frame()
    path = tmp_path / "drdid_binary_panel.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 17},
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
    assert result.metadata["outcome_scale"] == "risk_difference_att"
    assert result.metadata["binary_outcome"] is True
    assert {
        "drdid",
        "overlap_metric",
        "smd_max_unweighted",
        "smd_max_weighted",
    } <= set(result.metrics)
