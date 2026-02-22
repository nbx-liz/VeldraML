import pytest

from veldra.api import Artifact, evaluate, fit
from veldra.api.exceptions import VeldraValidationError


def _fit_binary_artifact(tmp_path, binary_frame):
    frame = binary_frame(rows=100, seed=12, coef1=1.4, coef2=-0.7, noise=0.3)
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


def test_binary_evaluate_returns_auc_logloss_brier(tmp_path, binary_frame) -> None:
    artifact, frame = _fit_binary_artifact(tmp_path, binary_frame)
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
        "top5_pct_positive",
    } <= set(result.metrics)
    assert result.metadata["n_rows"] == len(frame)
    assert result.metadata["target"] == "target"


def test_binary_evaluate_returns_phase35_metrics(tmp_path, binary_frame) -> None:
    artifact, frame = _fit_binary_artifact(tmp_path, binary_frame)
    result = evaluate(artifact, frame)
    assert "top5_pct_positive" in result.metrics
    assert 0.0 <= result.metrics["top5_pct_positive"] <= 1.0


def test_binary_evaluate_validation_errors(tmp_path, binary_frame) -> None:
    artifact, frame = _fit_binary_artifact(tmp_path, binary_frame)

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
                "task": {"type": "binary"},
                "data": {"target": "target"},
            },
            frame,
        )


def test_binary_evaluate_returns_precision_at_k_when_configured(tmp_path, binary_frame) -> None:
    frame = binary_frame(rows=90, seed=88, coef1=1.3, coef2=-0.6, noise=0.35)
    data_path = tmp_path / "train_binary_topk.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 21},
            "train": {"top_k": 10},
            "postprocess": {"calibration": "platt"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)
    result = evaluate(artifact, frame)

    assert "precision_at_10" in result.metrics
