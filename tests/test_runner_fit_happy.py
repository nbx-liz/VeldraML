from __future__ import annotations

from pathlib import Path

import pytest

from veldra.api import Artifact, fit
from veldra.api.types import RunResult


def _base_payload(task_type: str, data_path: Path, artifact_dir: Path) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": str(data_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 11},
        "train": {"num_boost_round": 40, "early_stopping_rounds": 10, "seed": 11},
        "export": {"artifact_dir": str(artifact_dir)},
    }
    if task_type in {"binary", "multiclass"}:
        payload["split"] = {"type": "stratified", "n_splits": 2, "seed": 11}
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
def test_fit_happy_path_all_task_types(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    task_type: str,
    frame_fixture: str,
) -> None:
    frame_builder = request.getfixturevalue(frame_fixture)
    if task_type == "multiclass":
        frame = frame_builder(rows_per_class=24, seed=11, scale=0.4)
    elif task_type == "frontier":
        frame = frame_builder(rows=90, seed=11)
    else:
        frame = frame_builder(rows=90, seed=11)

    data_path = tmp_path / f"{task_type}_train.csv"
    frame.to_csv(data_path, index=False)
    run = fit(_base_payload(task_type, data_path, tmp_path / "artifacts"))

    assert isinstance(run, RunResult)
    assert run.task_type == task_type
    assert run.run_id
    assert run.artifact_path
    assert run.metrics

    artifact_path = Path(run.artifact_path)
    assert artifact_path.exists()
    artifact = Artifact.load(artifact_path)
    assert artifact.run_config.task.type == task_type
