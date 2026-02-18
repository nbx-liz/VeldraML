from __future__ import annotations

from pathlib import Path

import pytest

from veldra.api import tune
from veldra.api.types import TuneResult


def _payload(task_type: str, data_path: Path, artifact_dir: Path) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": str(data_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 21},
        "train": {"num_boost_round": 30, "early_stopping_rounds": 8, "seed": 21},
        "tuning": {"enabled": True, "n_trials": 2, "preset": "fast", "log_level": "WARNING"},
        "export": {"artifact_dir": str(artifact_dir)},
    }
    if task_type in {"binary", "multiclass"}:
        payload["split"] = {"type": "stratified", "n_splits": 2, "seed": 21}
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
def test_tune_happy_path_minimum_trials(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    task_type: str,
    frame_fixture: str,
) -> None:
    frame_builder = request.getfixturevalue(frame_fixture)
    if task_type == "multiclass":
        frame = frame_builder(rows_per_class=18, seed=3, scale=0.5)
    elif task_type == "frontier":
        frame = frame_builder(rows=72, seed=3)
    else:
        frame = frame_builder(rows=72, seed=3)
    data_path = tmp_path / f"{task_type}_tune.csv"
    frame.to_csv(data_path, index=False)

    result = tune(_payload(task_type, data_path, tmp_path / "artifacts"))
    assert isinstance(result, TuneResult)
    assert result.task_type == task_type
    assert result.run_id
    assert result.best_params
    assert result.best_score is not None
    assert int(result.metadata["n_trials"]) == 2
    assert result.metadata["metric_name"]
