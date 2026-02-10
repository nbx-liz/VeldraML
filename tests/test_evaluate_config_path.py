from __future__ import annotations

import pandas as pd

from veldra.api import evaluate
from veldra.config.models import RunConfig


def test_evaluate_config_path_regression(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.2, 0.9, 0.7, 0.4, 0.2],
            "target": [0.1, 0.8, 1.9, 3.1, 3.8, 5.2],
        }
    )
    train_path = tmp_path / "regression_train.csv"
    train.to_csv(train_path, index=False)
    payload = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": str(train_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
        "export": {"artifact_dir": str(tmp_path)},
    }
    cfg = RunConfig.model_validate(payload)

    eval_frame = pd.DataFrame(
        {
            "x1": [0.5, 1.5, 2.5, 3.5],
            "x2": [1.1, 1.0, 0.8, 0.5],
            "target": [0.5, 1.6, 2.6, 3.6],
        }
    )
    result = evaluate(cfg, eval_frame)
    assert {"rmse", "mae", "r2"} <= set(result.metrics)
    assert result.metadata["evaluation_mode"] == "config"
    assert result.metadata["ephemeral_run"] is True
    assert result.metadata["train_source_path"] == str(train_path)
    assert result.metadata["artifact_run_id"].startswith("ephemeral_eval_")


def test_evaluate_config_path_binary(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "x1": [0.0, 0.3, 0.5, 1.0, 1.2, 1.5, 2.0, 2.5],
            "x2": [1.0, 0.9, 1.1, 0.6, 0.5, 0.4, 0.2, 0.1],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    train_path = tmp_path / "binary_train.csv"
    train.to_csv(train_path, index=False)
    payload = {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": str(train_path), "target": "target"},
        "split": {"type": "stratified", "n_splits": 2, "seed": 7},
        "postprocess": {"calibration": "platt"},
        "export": {"artifact_dir": str(tmp_path)},
    }
    result = evaluate(payload, train.copy())
    assert {"auc", "logloss", "brier", "accuracy", "f1", "precision", "recall"} <= set(
        result.metrics
    )
    assert result.metadata["evaluation_mode"] == "config"
    assert result.metadata["ephemeral_run"] is True
    assert result.metadata["train_source_path"] == str(train_path)


def test_evaluate_config_path_multiclass(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            "x2": [1.0, 0.9, 1.1, 0.2, 0.1, 0.3, -0.8, -0.9, -0.7],
            "target": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        }
    )
    train_path = tmp_path / "multiclass_train.csv"
    train.to_csv(train_path, index=False)
    payload = {
        "config_version": 1,
        "task": {"type": "multiclass"},
        "data": {"path": str(train_path), "target": "target"},
        "split": {"type": "stratified", "n_splits": 3, "seed": 7},
        "export": {"artifact_dir": str(tmp_path)},
    }
    result = evaluate(payload, train.copy())
    assert {"accuracy", "macro_f1", "logloss"} <= set(result.metrics)
    assert result.metadata["evaluation_mode"] == "config"
    assert result.metadata["ephemeral_run"] is True
    assert result.metadata["train_source_path"] == str(train_path)


def test_evaluate_config_path_frontier(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "x1": [0.0, 0.3, 0.8, 1.0, 1.2, 1.8, 2.0, 2.3],
            "x2": [1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.2, 0.1],
            "target": [0.2, 0.6, 1.1, 1.3, 1.6, 2.2, 2.5, 2.8],
        }
    )
    train_path = tmp_path / "frontier_train.csv"
    train.to_csv(train_path, index=False)
    payload = {
        "config_version": 1,
        "task": {"type": "frontier"},
        "data": {"path": str(train_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
        "frontier": {"alpha": 0.9},
        "export": {"artifact_dir": str(tmp_path)},
    }
    result = evaluate(payload, train.copy())
    assert {"pinball", "mae", "mean_u_hat", "coverage"} <= set(result.metrics)
    assert result.metadata["evaluation_mode"] == "config"
    assert result.metadata["ephemeral_run"] is True
    assert result.metadata["train_source_path"] == str(train_path)
