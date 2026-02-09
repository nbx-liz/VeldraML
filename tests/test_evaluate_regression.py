import pandas as pd
import pytest

from veldra.api import Artifact, VeldraNotImplementedError, VeldraValidationError, evaluate, fit
from veldra.config.models import RunConfig


def test_evaluate_regression_artifact_returns_metrics(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [float(i) for i in range(20)],
            "x2": [float(i % 4) for i in range(20)],
            "y": [float(i * 1.3 + (i % 4) * 0.5) for i in range(20)],
        }
    )
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "y"},
            "split": {"type": "kfold", "n_splits": 4, "seed": 5},
            "train": {"lgb_params": {"num_threads": 1}, "seed": 5},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)

    result = evaluate(artifact, frame)
    assert result.task_type == "regression"
    assert {"rmse", "mae", "r2"} == set(result.metrics.keys())
    assert result.metadata["target"] == "y"
    assert result.metadata["n_rows"] == len(frame)
    assert result.metadata["artifact_run_id"] == artifact.manifest.run_id


def test_evaluate_rejects_invalid_inputs(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0.1, 1.1, 2.1, 3.2, 4.3, 5.1],
        }
    )
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "y"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 3},
            "train": {"lgb_params": {"num_threads": 1}, "seed": 3},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)

    with pytest.raises(VeldraValidationError):
        evaluate(artifact, data=frame.drop(columns=["y"]))
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, data="not-a-dataframe")
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, data=frame.iloc[:0].copy())


def test_evaluate_rejects_non_artifact_or_non_regression() -> None:
    with pytest.raises(VeldraNotImplementedError):
        evaluate({"config_version": 1}, data=pd.DataFrame({"y": [1.0]}))

    binary_config = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"target": "y"},
        }
    )
    binary_artifact = Artifact.from_config(binary_config, run_id="binary-run")
    with pytest.raises(VeldraNotImplementedError):
        evaluate(binary_artifact, data=pd.DataFrame({"y": [1.0]}))
