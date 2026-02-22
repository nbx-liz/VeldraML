import pytest

from veldra.api import Artifact, evaluate, fit
from veldra.api.exceptions import VeldraValidationError


def _fit_regression_artifact(tmp_path, regression_frame) -> tuple[Artifact, object]:
    frame = regression_frame(rows=100, seed=12)
    data_path = tmp_path / "train_regression.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 21},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    return Artifact.load(run.artifact_path), frame


def test_regression_evaluate_returns_rmse_mae_r2(tmp_path, regression_frame) -> None:
    artifact, frame = _fit_regression_artifact(tmp_path, regression_frame)
    result = evaluate(artifact, frame)

    assert result.task_type == "regression"
    assert {"rmse", "mae", "r2", "huber"} <= set(result.metrics)
    assert result.metadata["n_rows"] == len(frame)
    assert result.metadata["target"] == "target"


def test_regression_evaluate_returns_phase35_metrics(tmp_path, regression_frame) -> None:
    artifact, frame = _fit_regression_artifact(tmp_path, regression_frame)
    result = evaluate(artifact, frame)
    assert "huber" in result.metrics
    assert result.metrics["huber"] >= 0.0


def test_regression_evaluate_validation_errors(tmp_path, regression_frame) -> None:
    artifact, frame = _fit_regression_artifact(tmp_path, regression_frame)

    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame.drop(columns=["target"]))
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, data=None)
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame.iloc[0:0])

    with pytest.raises(VeldraValidationError):
        evaluate(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"target": "target"},
            },
            frame,
        )
