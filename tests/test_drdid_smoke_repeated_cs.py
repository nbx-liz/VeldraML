from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import estimate_dr


def _repeated_cs_frame(n_pre: int = 120, n_post: int = 120, seed: int = 222) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    post = np.concatenate([np.zeros(n_pre, dtype=int), np.ones(n_post, dtype=int)])
    age = rng.integers(20, 60, size=n)
    skill = rng.normal(size=n)
    logits = -0.6 + 0.05 * (age - 30) + 0.6 * skill + 0.1 * post
    p = 1.0 / (1.0 + np.exp(-logits))
    treatment = rng.binomial(1, p, size=n)
    outcome = (
        9000
        + 260 * (age - 30)
        + 1400 * skill
        + 700 * post
        + 1500 * treatment * post
        + rng.normal(0, 1200, size=n)
    )
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


def test_estimate_drdid_repeated_cross_section_smoke(tmp_path) -> None:
    frame = _repeated_cs_frame()
    path = tmp_path / "repeated.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 9},
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
    assert {"naive", "ipw", "dr"} <= set(result.metrics)

