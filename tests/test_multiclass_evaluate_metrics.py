import pytest

from veldra.api import Artifact, evaluate, fit
from veldra.api.exceptions import VeldraValidationError


def _train_artifact(tmp_path, multiclass_frame) -> Artifact:
    frame = multiclass_frame(rows_per_class=30, seed=31, scale=0.5)
    data_path = tmp_path / "multiclass_train.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 2},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    return Artifact.load(run.artifact_path)


def test_multiclass_evaluate_returns_expected_metrics(tmp_path, multiclass_frame) -> None:
    artifact = _train_artifact(tmp_path, multiclass_frame)
    frame = multiclass_frame(rows_per_class=12, seed=32, scale=0.5)
    result = evaluate(artifact, frame)
    assert {
        "accuracy",
        "macro_f1",
        "logloss",
        "balanced_accuracy",
        "brier_macro",
        "ovr_roc_auc_macro",
        "average_precision_macro",
    } <= set(result.metrics.keys())
    assert result.task_type == "multiclass"


def test_multiclass_evaluate_returns_phase35_metrics(tmp_path, multiclass_frame) -> None:
    artifact = _train_artifact(tmp_path, multiclass_frame)
    frame = multiclass_frame(rows_per_class=12, seed=32, scale=0.5)
    result = evaluate(artifact, frame)
    assert {
        "balanced_accuracy",
        "brier_macro",
        "ovr_roc_auc_macro",
        "average_precision_macro",
    } <= set(result.metrics)


def test_multiclass_evaluate_continues_when_eval_data_is_missing_a_class(
    tmp_path, multiclass_frame
) -> None:
    artifact = _train_artifact(tmp_path, multiclass_frame)
    frame = multiclass_frame(rows_per_class=12, seed=32, scale=0.5)
    frame_missing_class = frame[frame["target"] != "gamma"].reset_index(drop=True)
    result = evaluate(artifact, frame_missing_class)

    assert {"accuracy", "macro_f1", "logloss"} <= set(result.metrics)
    assert {"balanced_accuracy", "brier_macro"} <= set(result.metrics)
    # Missing-class inputs may make one-vs-rest aggregates undefined; evaluate() must still succeed.
    assert "ovr_roc_auc_macro" not in result.metrics


def test_multiclass_evaluate_validation_errors(tmp_path, multiclass_frame) -> None:
    artifact = _train_artifact(tmp_path, multiclass_frame)
    frame = multiclass_frame(rows_per_class=8, seed=33, scale=0.5)

    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame.drop(columns=["target"]))
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame.iloc[0:0])
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, data="not-a-dataframe")


def test_multiclass_evaluate_config_input_validation(tmp_path, multiclass_frame) -> None:
    artifact = _train_artifact(tmp_path, multiclass_frame)
    frame = multiclass_frame(rows_per_class=5, seed=34, scale=0.5)
    _ = artifact
    with pytest.raises(VeldraValidationError):
        evaluate({"task": {"type": "multiclass"}}, frame)
