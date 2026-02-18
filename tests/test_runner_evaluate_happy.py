from __future__ import annotations

from pathlib import Path

import pytest

from veldra.api import Artifact, evaluate, fit
from veldra.api.types import EvalResult


def _payload(task_type: str, data_path: Path, artifact_dir: Path) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": str(data_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
        "train": {"num_boost_round": 35, "early_stopping_rounds": 8, "seed": 7},
        "export": {"artifact_dir": str(artifact_dir)},
    }
    if task_type in {"binary", "multiclass"}:
        payload["split"] = {"type": "stratified", "n_splits": 2, "seed": 7}
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
def test_evaluate_happy_path_artifact_and_config(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    task_type: str,
    frame_fixture: str,
) -> None:
    frame_builder = request.getfixturevalue(frame_fixture)
    if task_type == "multiclass":
        frame = frame_builder(rows_per_class=22, seed=5, scale=0.45)
    elif task_type == "frontier":
        frame = frame_builder(rows=84, seed=5)
    else:
        frame = frame_builder(rows=84, seed=5)

    train_path = tmp_path / f"{task_type}_train.csv"
    frame.to_csv(train_path, index=False)
    payload = _payload(task_type, train_path, tmp_path / "artifacts")
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    eval_frame = frame.sample(frac=0.35, random_state=9).reset_index(drop=True)

    result_from_artifact = evaluate(artifact, eval_frame)
    assert isinstance(result_from_artifact, EvalResult)
    assert result_from_artifact.task_type == task_type
    assert result_from_artifact.metrics
    assert result_from_artifact.metadata["evaluation_mode"] == "artifact"
    assert result_from_artifact.metadata["ephemeral_run"] is False

    result_from_config = evaluate(payload, eval_frame)
    assert isinstance(result_from_config, EvalResult)
    assert result_from_config.task_type == task_type
    assert result_from_config.metrics
    assert result_from_config.metadata["evaluation_mode"] == "config"
    assert result_from_config.metadata["ephemeral_run"] is True
    assert result_from_config.metadata["train_source_path"] == str(train_path)
