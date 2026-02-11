from __future__ import annotations

import pandas as pd

from veldra.api import estimate_dr


def test_estimate_dr_smoke_regression(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 0.4, 0.7, 0.9, 1.2, 1.4, 1.6, 1.8],
            "x2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2],
            "treatment": [0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            "outcome": [1.0, 1.1, 1.2, 2.0, 1.4, 2.5, 2.6, 2.8, 3.0, 1.9],
        }
    )
    train_path = tmp_path / "dr_reg_train.csv"
    frame.to_csv(train_path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(train_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "causal": {"treatment_col": "treatment"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )

    assert result.method == "dr"
    assert result.estimand == "att"
    assert isinstance(result.estimate, float)
    assert {"naive", "ipw", "dr"} <= set(result.metrics)


def test_estimate_dr_smoke_binary_outcome(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 0.4, 0.7, 0.9, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
            "x2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "treatment": [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            "outcome": [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
        }
    )
    train_path = tmp_path / "dr_bin_train.csv"
    frame.to_csv(train_path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(train_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 19},
            "causal": {"treatment_col": "treatment", "estimand": "ate"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    assert result.estimand == "ate"
    assert result.std_error is not None
