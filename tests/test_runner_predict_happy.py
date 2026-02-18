from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from veldra.api import Artifact, fit, predict
from veldra.api.types import Prediction


def _payload(task_type: str, data_path: Path, artifact_dir: Path) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": str(data_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 3},
        "train": {"num_boost_round": 35, "early_stopping_rounds": 8, "seed": 3},
        "export": {"artifact_dir": str(artifact_dir)},
    }
    if task_type in {"binary", "multiclass"}:
        payload["split"] = {"type": "stratified", "n_splits": 2, "seed": 3}
    if task_type == "binary":
        payload["postprocess"] = {"calibration": "platt"}
    if task_type == "frontier":
        payload["frontier"] = {"alpha": 0.9}
    return payload


@pytest.mark.parametrize(
    ("task_type", "frame_fixture"),
    [
        ("regression", "regression_frame"),
        ("binary", "binary_frame"),
        ("multiclass", "multiclass_frame"),
        ("frontier", "frontier_frame"),
    ],
)
def test_predict_happy_path_all_task_types(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    task_type: str,
    frame_fixture: str,
) -> None:
    frame_builder = request.getfixturevalue(frame_fixture)
    if task_type == "multiclass":
        frame = frame_builder(rows_per_class=20, seed=8, scale=0.4)
    elif task_type == "frontier":
        frame = frame_builder(rows=80, seed=8)
    else:
        frame = frame_builder(rows=80, seed=8)

    train_path = tmp_path / f"{task_type}_train.csv"
    frame.to_csv(train_path, index=False)
    run = fit(_payload(task_type, train_path, tmp_path / "artifacts"))
    artifact = Artifact.load(run.artifact_path)

    x = frame.drop(columns=["target"]).head(16).reset_index(drop=True)
    pred = predict(artifact, x)

    assert isinstance(pred, Prediction)
    assert pred.task_type == task_type
    assert pred.metadata["n_rows"] == len(x)

    if task_type == "regression":
        assert len(pred.data) == len(x)
    elif task_type == "binary":
        assert isinstance(pred.data, pd.DataFrame)
        assert {"p_cal", "p_raw", "label_pred"} <= set(pred.data.columns)
    elif task_type == "multiclass":
        assert isinstance(pred.data, pd.DataFrame)
        assert "label_pred" in pred.data.columns
        assert any(c.startswith("proba_") for c in pred.data.columns)
    else:
        assert isinstance(pred.data, pd.DataFrame)
        assert "frontier_pred" in pred.data.columns
