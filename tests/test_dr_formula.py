from __future__ import annotations

import pandas as pd

from veldra.api import estimate_dr


def _synthetic_dr_frame(n: int = 900, seed: int = 123) -> tuple[pd.DataFrame, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treat_logit = 0.4 * x1 - 0.5 * x2 + 0.15
    treat_prob = 1.0 / (1.0 + np.exp(-treat_logit))
    treatment = (rng.uniform(size=n) < treat_prob).astype(int)
    true_tau = 1.2 + 0.2 * x1
    baseline = 1.5 + 0.8 * x1 - 0.7 * x2
    noise = rng.normal(scale=0.3, size=n)
    outcome = baseline + true_tau * treatment + noise

    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "treatment": treatment,
            "outcome": outcome,
            "true_tau": true_tau,
        }
    )
    true_att = float(frame.loc[frame["treatment"] == 1, "true_tau"].mean())
    return frame, true_att


def test_dr_estimate_is_reasonably_close_to_ground_truth_att(tmp_path) -> None:
    frame, true_att = _synthetic_dr_frame(n=900, seed=123)
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(train_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 17},
            "causal": {"treatment_col": "treatment", "estimand": "att"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert abs(result.estimate - true_att) < 0.9
    assert "dr" in result.metrics


def test_dr_observation_contains_psi_and_weights(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 0.4, 0.7, 0.9, 1.2, 1.4],
            "x2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "treatment": [0, 0, 0, 1, 0, 1, 1, 1],
            "outcome": [1.0, 1.1, 1.2, 2.0, 1.4, 2.5, 2.6, 2.8],
        }
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 3},
            "causal": {"treatment_col": "treatment"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    obs = pd.read_parquet(result.metadata["observation_path"])
    assert {"psi", "weight", "m0_hat", "m1_hat"} <= set(obs.columns)
