from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veldra.api import fit
from veldra.api.exceptions import VeldraValidationError


def test_fit_rejects_empty_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["x1", "x2", "target"]).to_csv(path, index=False)
    with pytest.raises(VeldraValidationError, match="Input data is empty"):
        fit(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 1},
                "export": {"artifact_dir": str(tmp_path / "artifacts")},
            }
        )


def test_fit_single_row_fails_with_invalid_split(tmp_path: Path) -> None:
    path = tmp_path / "single_row.csv"
    pd.DataFrame({"x1": [1.0], "x2": [2.0], "target": [3.0]}).to_csv(path, index=False)
    with pytest.raises(Exception):
        fit(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 1},
                "export": {"artifact_dir": str(tmp_path / "artifacts")},
            }
        )


def test_fit_accepts_constant_target_regression(tmp_path: Path) -> None:
    rows = 36
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, rows),
            "x2": np.linspace(1.0, 2.0, rows),
            "target": np.ones(rows) * 7.5,
        }
    )
    path = tmp_path / "constant_target.csv"
    frame.to_csv(path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 1},
            "train": {"num_boost_round": 25, "early_stopping_rounds": 6},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert run.task_type == "regression"
    assert run.metrics


def test_fit_rejects_nan_target_values(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4],
            "x2": [0.0, 1.0, 0.5, 0.8],
            "target": [1.0, np.nan, 1.5, 2.0],
        }
    )
    path = tmp_path / "nan_target.csv"
    frame.to_csv(path, index=False)
    with pytest.raises((VeldraValidationError, ValueError), match="null values|contains NaN"):
        fit(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 1},
                "export": {"artifact_dir": str(tmp_path / "artifacts")},
            }
        )


def test_fit_handles_extreme_class_imbalance(
    tmp_path: Path, unbalanced_binary_frame: pd.DataFrame
) -> None:
    path = tmp_path / "unbalanced.csv"
    unbalanced_binary_frame.to_csv(path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 13},
            "postprocess": {"calibration": "platt"},
            "train": {"num_boost_round": 30, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert run.task_type == "binary"
    assert run.metrics
