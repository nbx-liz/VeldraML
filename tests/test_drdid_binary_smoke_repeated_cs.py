from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import estimate_dr


def _repeated_cs_binary_frame(n_pre: int = 180, n_post: int = 180, seed: int = 902) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_pre + n_post
    post = np.concatenate([np.zeros(n_pre, dtype=int), np.ones(n_post, dtype=int)])
    age = rng.integers(20, 60, size=n_total)
    skill = rng.normal(size=n_total)
    logits_t = -0.8 + 0.04 * (age - 30) + 0.7 * skill + 0.15 * post
    p_t = 1.0 / (1.0 + np.exp(-logits_t))
    treatment = rng.binomial(1, p_t, size=n_total)

    logits_y = -1.6 + 0.03 * (age - 30) + 0.5 * skill + 0.2 * post + 0.6 * treatment * post
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits_y)), size=n_total)
    return pd.DataFrame(
        {
            "time": post,
            "post": post,
            "treatment": treatment,
            "age": age,
            "skill": skill,
            "outcome": outcome,
        }
    )


def test_estimate_drdid_binary_repeated_cross_section_smoke(tmp_path) -> None:
    frame = _repeated_cs_binary_frame()
    path = tmp_path / "drdid_binary_repeated.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 19},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "repeated_cross_section",
                "time_col": "time",
                "post_col": "post",
            },
            "export": {"artifact_dir": str(tmp_path)},
        }
    )

    assert result.method == "dr_did"
    assert result.metadata["design"] == "repeated_cross_section"
    assert result.metadata["outcome_scale"] == "risk_difference_att"
    assert result.metadata["binary_outcome"] is True
    assert {
        "drdid",
        "overlap_metric",
        "smd_max_unweighted",
        "smd_max_weighted",
    } <= set(result.metrics)
