import numpy as np
import pandas as pd
import pytest

from veldra.api import Artifact, evaluate, fit
from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError


def _binary_frame(rows: int = 100, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 1.4 * x1 - 0.7 * x2 + rng.normal(scale=0.3, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _fit_binary_artifact(tmp_path) -> tuple[Artifact, pd.DataFrame]:
    frame = _binary_frame()
    data_path = tmp_path / "train_binary.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 21},
            "postprocess": {"calibration": "platt"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    return Artifact.load(run.artifact_path), frame


def test_binary_evaluate_returns_auc_logloss_brier(tmp_path) -> None:
    artifact, frame = _fit_binary_artifact(tmp_path)
    result = evaluate(artifact, frame)

    assert result.task_type == "binary"
    assert {
        "auc",
        "logloss",
        "brier",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "threshold",
    } <= set(result.metrics)
    assert result.metadata["n_rows"] == len(frame)
    assert result.metadata["target"] == "target"


def test_binary_evaluate_validation_errors(tmp_path) -> None:
    artifact, frame = _fit_binary_artifact(tmp_path)

    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame.drop(columns=["target"]))
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, data=None)
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame.iloc[0:0])

    with pytest.raises(VeldraNotImplementedError):
        evaluate(
            {
                "config_version": 1,
                "task": {"type": "binary"},
                "data": {"target": "target"},
            },
            frame,
        )
