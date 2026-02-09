import pandas as pd
import pytest

from veldra.api import (
    Artifact,
    VeldraNotImplementedError,
    VeldraValidationError,
    evaluate,
    export,
    fit,
    predict,
    simulate,
    tune,
)


def _config_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "", "target": "y"},
    }


def test_api_symbols_are_importable() -> None:
    assert callable(fit)
    assert callable(tune)
    assert callable(evaluate)
    assert callable(predict)
    assert callable(simulate)
    assert callable(export)
    assert hasattr(Artifact, "load")
    assert hasattr(Artifact, "save")
    assert hasattr(Artifact, "predict")
    assert hasattr(Artifact, "simulate")


def test_unimplemented_runner_endpoints_raise_consistent_error(tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "y": [0.2, 1.1, 1.9, 3.2]})
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)

    payload = _config_payload()
    payload["data"]["path"] = str(data_path)
    payload["split"] = {"type": "kfold", "n_splits": 2, "seed": 7}
    payload["export"] = {"artifact_dir": str(tmp_path)}
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    with pytest.raises(VeldraNotImplementedError):
        tune(payload)
    with pytest.raises(VeldraNotImplementedError):
        evaluate(payload, data=None)
    with pytest.raises(VeldraValidationError):
        predict(artifact, data=None)
    pred = predict(artifact, data=frame[["x1"]])
    assert len(pred.data) == len(frame)
    with pytest.raises(VeldraNotImplementedError):
        simulate(artifact, data=None, scenarios=None)
    with pytest.raises(VeldraNotImplementedError):
        export(artifact, format="python")


def test_binary_predict_and_evaluate_paths_are_implemented(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 0.8, 1.2, 0.7, 1.1, 0.9],
            "y": [0, 0, 0, 1, 1, 1],
        }
    )
    data_path = tmp_path / "binary_train.csv"
    frame.to_csv(data_path, index=False)

    payload = {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": str(data_path), "target": "y"},
        "split": {"type": "stratified", "n_splits": 2, "seed": 1},
        "postprocess": {"calibration": "platt"},
        "export": {"artifact_dir": str(tmp_path)},
    }
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    pred = predict(artifact, data=frame[["x1", "x2"]])
    assert list(pred.data.columns) == ["p_cal", "p_raw", "label_pred"]

    eval_result = evaluate(artifact, frame)
    assert {"auc", "logloss", "brier"} <= set(eval_result.metrics)


def test_multiclass_predict_and_evaluate_paths_are_implemented(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.0, 2.1, 2.3, 2.0, 4.0, 4.2, 3.9],
            "x2": [1.0, 1.1, 0.9, 0.2, 0.1, 0.0, -0.8, -0.9, -1.0],
            "y": [
                "alpha",
                "alpha",
                "alpha",
                "beta",
                "beta",
                "beta",
                "gamma",
                "gamma",
                "gamma",
            ],
        }
    )
    data_path = tmp_path / "multiclass_train.csv"
    frame.to_csv(data_path, index=False)

    payload = {
        "config_version": 1,
        "task": {"type": "multiclass"},
        "data": {"path": str(data_path), "target": "y"},
        "split": {"type": "stratified", "n_splits": 3, "seed": 1},
        "export": {"artifact_dir": str(tmp_path)},
    }
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    pred = predict(artifact, data=frame[["x1", "x2"]])
    assert "label_pred" in pred.data.columns
    assert {"proba_alpha", "proba_beta", "proba_gamma"} <= set(pred.data.columns)

    eval_result = evaluate(artifact, frame)
    assert {"accuracy", "macro_f1", "logloss"} <= set(eval_result.metrics)
