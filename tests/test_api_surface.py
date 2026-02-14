import pandas as pd
import pytest

from veldra.api import (
    Artifact,
    VeldraValidationError,
    estimate_dr,
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
    assert callable(estimate_dr)
    assert hasattr(Artifact, "load")
    assert hasattr(Artifact, "save")
    assert hasattr(Artifact, "predict")
    assert hasattr(Artifact, "simulate")


def test_runner_endpoints_raise_consistent_error_for_unimplemented_only(tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "y": [0.2, 1.1, 1.9, 3.2]})
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)

    payload = _config_payload()
    payload["data"]["path"] = str(data_path)
    payload["split"] = {"type": "kfold", "n_splits": 2, "seed": 7}
    payload["export"] = {"artifact_dir": str(tmp_path)}
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    tune_payload = dict(payload)
    tune_payload["tuning"] = {"enabled": True, "n_trials": 1, "preset": "fast"}
    tune_result = tune(tune_payload)
    assert tune_result.best_score is not None
    assert tune_result.best_params
    with pytest.raises(VeldraValidationError):
        evaluate(payload, data=None)
    with pytest.raises(VeldraValidationError):
        predict(artifact, data=None)
    pred = predict(artifact, data=frame[["x1"]])
    assert len(pred.data) == len(frame)
    with pytest.raises(VeldraValidationError):
        simulate(artifact, data=None, scenarios=None)
    sim = simulate(
        artifact,
        data=frame,
        scenarios={"name": "x1_shift", "actions": [{"op": "add", "column": "x1", "value": 0.2}]},
    )
    assert len(sim.data) == len(frame)
    assert {"row_id", "scenario", "task_type"} <= set(sim.data.columns)
    exported = export(artifact, format="python")
    assert exported.format == "python"
    assert exported.path


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


def test_frontier_predict_and_evaluate_paths_are_implemented(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.0, 2.1, 2.3, 2.0, 4.0, 4.2, 3.9],
            "x2": [1.0, 1.1, 0.9, 0.2, 0.1, 0.0, -0.8, -0.9, -1.0],
            "y": [1.2, 1.1, 1.4, 3.5, 3.4, 3.8, 5.0, 5.3, 5.1],
        }
    )
    data_path = tmp_path / "frontier_train.csv"
    frame.to_csv(data_path, index=False)

    payload = {
        "config_version": 1,
        "task": {"type": "frontier"},
        "data": {"path": str(data_path), "target": "y"},
        "split": {"type": "kfold", "n_splits": 3, "seed": 1},
        "frontier": {"alpha": 0.90},
        "export": {"artifact_dir": str(tmp_path)},
    }
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    pred = predict(artifact, data=frame[["x1", "x2"]])
    assert "frontier_pred" in pred.data.columns

    eval_result = evaluate(artifact, frame)
    assert {"pinball", "mae", "mean_u_hat", "coverage"} <= set(eval_result.metrics)

    tune_result = tune(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "y"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 2},
            "frontier": {"alpha": 0.90},
            "tuning": {"enabled": True, "n_trials": 1, "preset": "fast"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    assert tune_result.metadata["metric_name"] == "pinball"


def test_estimate_dr_is_implemented(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 1.8],
            "x2": [1.0, 0.9, 1.1, 0.8, 0.7, 0.6, 0.4, 0.3],
            "treatment": [0, 0, 1, 0, 1, 1, 1, 0],
            "outcome": [1.0, 1.2, 2.2, 1.4, 2.5, 2.8, 3.0, 1.7],
        }
    )
    data_path = tmp_path / "dr_train.csv"
    frame.to_csv(data_path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 11},
            "causal": {"treatment_col": "treatment"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    assert result.method == "dr"
    assert result.estimand == "att"
    assert "dr" in result.metrics
    assert {"overlap_metric", "smd_max_unweighted", "smd_max_weighted"} <= set(result.metrics)

    panel = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "treatment": [0, 0, 1, 1, 0, 0, 1, 1],
            "x1": [0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4],
            "outcome": [10.0, 10.9, 12.0, 14.0, 9.7, 10.5, 13.2, 15.0],
        }
    )
    panel_path = tmp_path / "drdid_panel.csv"
    panel.to_csv(panel_path, index=False)
    did_result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(panel_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 11},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
            },
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    assert did_result.method == "dr_did"
    assert did_result.metadata["design"] == "panel"
    assert {"overlap_metric", "smd_max_unweighted", "smd_max_weighted"} <= set(did_result.metrics)

    panel_binary = panel.copy()
    panel_binary["outcome"] = [0, 1, 0, 1, 0, 0, 1, 1]
    panel_binary_path = tmp_path / "drdid_panel_binary.csv"
    panel_binary.to_csv(panel_binary_path, index=False)
    did_binary_result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(panel_binary_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 11},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
            },
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    assert did_binary_result.method == "dr_did"
    assert did_binary_result.metadata["binary_outcome"] is True
