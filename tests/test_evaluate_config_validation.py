from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import evaluate
from veldra.api.exceptions import VeldraValidationError


def _base_payload(train_path: str) -> dict:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": train_path, "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
        "export": {"artifact_dir": "."},
    }


def test_evaluate_config_requires_data_path(tmp_path) -> None:
    payload = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
    }
    eval_frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0.2, 0.3]})
    with pytest.raises(VeldraValidationError, match="data.path is required"):
        evaluate(payload, eval_frame)


def test_evaluate_config_rejects_non_dataframe_input(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "target": [0.1, 0.9, 2.1, 2.9],
        }
    )
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)
    payload = _base_payload(str(train_path))

    with pytest.raises(VeldraValidationError, match="pandas.DataFrame"):
        evaluate(payload, None)


def test_evaluate_config_rejects_missing_target_column(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "target": [0.1, 0.9, 2.1, 2.9],
        }
    )
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)
    payload = _base_payload(str(train_path))

    with pytest.raises(VeldraValidationError, match="missing target column"):
        evaluate(payload, pd.DataFrame({"x1": [0.2, 0.8]}))


def test_evaluate_config_rejects_empty_dataframe(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "target": [0.1, 0.9, 2.1, 2.9],
        }
    )
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)
    payload = _base_payload(str(train_path))

    empty = pd.DataFrame(columns=["x1", "target"])
    with pytest.raises(VeldraValidationError, match="DataFrame is empty"):
        evaluate(payload, empty)
