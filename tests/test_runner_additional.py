from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.api import Artifact, evaluate, export, fit, predict, simulate
from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError
from veldra.api.runner import tune
from veldra.config.models import RunConfig


def _config_payload(task_type: str, path: str | None = None, target: str = "target") -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"target": target},
    }
    if path is not None:
        payload["data"]["path"] = path
    return payload


def _artifact_for_task(task_type: str, feature_schema: dict | None = None) -> Artifact:
    cfg = RunConfig.model_validate(_config_payload(task_type, path="dummy.csv"))
    return Artifact.from_config(
        run_config=cfg,
        run_id=f"rid_{task_type}",
        feature_schema=feature_schema or {"feature_names": ["x1"]},
    )


def test_ensure_config_passthrough_instance() -> None:
    cfg = RunConfig.model_validate(_config_payload("regression", path="dummy.csv"))
    from veldra.api.runner import _ensure_config

    assert _ensure_config(cfg) is cfg


def test_fit_validation_and_notimplemented_paths(tmp_path) -> None:
    with pytest.raises(VeldraValidationError):
        fit({})

    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "target": [0.1, 0.4, 0.9, 1.2, 1.6, 2.0],
        }
    )
    data_path = tmp_path / "frontier.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    assert run.task_type == "frontier"

    with pytest.raises(VeldraValidationError):
        fit(_config_payload("regression", path=None))


def test_simulate_implemented_and_export_unimplemented() -> None:
    artifact = _artifact_for_task("regression")
    artifact.predict = lambda df: np.asarray(df["x1"], dtype=float)  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        simulate(artifact, data=None, scenarios=None)
    simulated = simulate(
        artifact,
        data=pd.DataFrame({"x1": [1.0, 2.0], "target": [1.0, 2.0]}),
        scenarios={"name": "scale", "actions": [{"op": "mul", "column": "x1", "value": 2.0}]},
    )
    assert {"row_id", "scenario", "task_type", "base_pred", "scenario_pred", "delta_pred"} <= set(
        simulated.data.columns
    )
    with pytest.raises(VeldraValidationError):
        export(artifact, format="python")


def test_runner_predict_rejects_unsupported_task() -> None:
    artifact = _artifact_for_task("frontier")
    artifact.run_config.task.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraNotImplementedError):
        predict(artifact, pd.DataFrame({"x1": [1.0]}))


def test_evaluate_rejects_unsupported_task_type() -> None:
    artifact = _artifact_for_task("frontier")
    artifact.run_config.task.type = "unknown"  # type: ignore[assignment]
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0, 1]})
    with pytest.raises(VeldraNotImplementedError):
        evaluate(artifact, frame)


def test_binary_evaluate_error_branches() -> None:
    artifact = _artifact_for_task(
        "binary",
        feature_schema={"feature_names": ["x1"], "target_classes": [0, 1]},
    )
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0, 1]})

    artifact.predict = lambda df: np.array([0.2, 0.3])  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)

    artifact.predict = lambda df: pd.DataFrame({"p_raw": [0.2, 0.3]})  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)

    artifact.feature_schema = {"feature_names": ["x1"]}
    artifact.predict = lambda df: pd.DataFrame({"p_cal": [0.2, 0.3]})  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)

    artifact.feature_schema = {"feature_names": ["x1"], "target_classes": [0, 1]}
    frame_outside = pd.DataFrame({"x1": [0.1, 0.2], "target": [2, 2]})
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame_outside)

    frame_one_class = pd.DataFrame({"x1": [0.1, 0.2], "target": [1, 1]})
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame_one_class)


def test_multiclass_evaluate_error_branches() -> None:
    artifact = _artifact_for_task(
        "multiclass",
        feature_schema={"feature_names": ["x1"], "target_classes": ["a", "b", "c"]},
    )
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": ["a", "b"]})

    artifact.predict = lambda df: np.array([0.1, 0.2])  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)

    artifact.predict = lambda df: pd.DataFrame({"label_pred": ["a", "b"]})  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)

    artifact.predict = lambda df: pd.DataFrame(  # type: ignore[method-assign]
        {
            "proba_a": [0.2, 0.4],
            "proba_b": [0.3, 0.3],
            "proba_c": [0.5, 0.3],
            "label_pred": ["c", "a"],
        }
    )
    frame_outside = pd.DataFrame({"x1": [0.1, 0.2], "target": ["z", "z"]})
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame_outside)

    artifact.feature_schema = {"feature_names": ["x1"]}
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)


def test_fit_and_tune_unsupported_task_branches(tmp_path) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(tmp_path / "dummy.csv"), "target": "target"},
        }
    )
    cfg.task.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraNotImplementedError):
        fit(cfg)

    cfg_tune = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(tmp_path / "dummy.csv"), "target": "target"},
            "tuning": {"enabled": True, "n_trials": 1},
        }
    )
    cfg_tune.task.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraValidationError):
        tune(cfg_tune)


def test_frontier_evaluate_error_branches() -> None:
    artifact = _artifact_for_task("frontier", feature_schema={"feature_names": ["x1"]})
    frame = pd.DataFrame({"x1": [0.1, 0.2], "target": [0.0, 1.0]})

    artifact.predict = lambda df: np.array([0.2, 0.3])  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)

    artifact.predict = lambda df: pd.DataFrame({"wrong_col": [0.2, 0.3]})  # type: ignore[method-assign]
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame)

    artifact.predict = lambda df: pd.DataFrame({"frontier_pred": [0.2, 0.3]})  # type: ignore[method-assign]
    frame_bad_target = pd.DataFrame({"x1": [0.1, 0.2], "target": ["a", "b"]})
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame_bad_target)
