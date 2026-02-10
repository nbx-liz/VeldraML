from __future__ import annotations

import pandas as pd

from veldra.api import estimate_dr


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 0.4, 0.7, 0.9, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
            "x2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "treatment": [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            "outcome": [1.0, 1.1, 1.2, 2.0, 1.4, 2.5, 2.6, 2.8, 3.0, 1.9, 3.2, 2.0],
        }
    )


def test_dr_uses_calibrated_propensity_with_platt(tmp_path) -> None:
    frame = _base_frame()
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "causal": {
                "treatment_col": "treatment",
                "propensity_calibration": "platt",
                "propensity_clip": 0.05,
            },
            "export": {"artifact_dir": str(tmp_path)},
        }
    )

    obs = pd.read_parquet(result.metadata["observation_path"])
    assert {"e_raw", "e_hat"} <= set(obs.columns)
    assert ((obs["e_hat"] >= 0.05) & (obs["e_hat"] <= 0.95)).all()


def test_dr_supports_isotonic_calibration(tmp_path) -> None:
    frame = _base_frame()
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "causal": {
                "treatment_col": "treatment",
                "propensity_calibration": "isotonic",
            },
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    assert result.metadata["summary_path"]
