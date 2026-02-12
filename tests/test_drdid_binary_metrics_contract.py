from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import estimate_dr


def _panel_frame(task: str, seed: int = 444) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_units = 90
    age = rng.integers(20, 58, size=n_units)
    skill = rng.normal(size=n_units)
    p_t = 1.0 / (1.0 + np.exp(-(-0.7 + 0.03 * (age - 30) + 0.8 * skill)))
    treatment = rng.binomial(1, p_t, size=n_units)

    if task == "binary":
        logits_pre = -1.4 + 0.02 * (age - 30) + 0.5 * skill
        logits_post = logits_pre + 0.2 + 0.6 * treatment
        y_pre = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits_pre)), size=n_units)
        y_post = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits_post)), size=n_units)
    else:
        base = 9000 + 220 * (age - 30) + 1500 * skill + rng.normal(0, 900, size=n_units)
        y_pre = base + rng.normal(0, 500, size=n_units)
        y_post = base + 700 + treatment * 1300 + rng.normal(0, 500, size=n_units)

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


def _estimate(path: str, task: str, out_dir: str):
    return estimate_dr(
        {
            "config_version": 1,
            "task": {"type": task},
            "data": {"path": path, "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 31},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
            },
            "export": {"artifact_dir": out_dir},
        }
    )


def test_drdid_metrics_contract_binary_and_regression(tmp_path) -> None:
    binary_df = _panel_frame("binary")
    binary_path = tmp_path / "binary_panel.csv"
    binary_df.to_csv(binary_path, index=False)
    binary_result = _estimate(str(binary_path), "binary", str(tmp_path))

    assert binary_result.metadata["outcome_scale"] == "risk_difference_att"
    assert binary_result.metadata["binary_outcome"] is True
    assert {"drdid", "overlap_metric", "smd_max_unweighted", "smd_max_weighted"} <= set(
        binary_result.metrics
    )

    regression_df = _panel_frame("regression")
    regression_path = tmp_path / "regression_panel.csv"
    regression_df.to_csv(regression_path, index=False)
    regression_result = _estimate(str(regression_path), "regression", str(tmp_path))

    assert regression_result.metadata["binary_outcome"] is False
    assert regression_result.metadata["outcome_scale"] is None
    assert {"naive", "ipw", "dr", "drdid"} <= set(regression_result.metrics)
    assert {"overlap_metric", "smd_max_unweighted", "smd_max_weighted"} <= set(
        regression_result.metrics
    )
