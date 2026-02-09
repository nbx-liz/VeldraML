from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import Artifact, evaluate, fit


def _binary_frame(rows: int = 180, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 0.9 * x1 - 0.8 * x2 + rng.normal(scale=0.8, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_binary_threshold_optimization_is_opt_in(tmp_path) -> None:
    frame = _binary_frame()
    data_path = tmp_path / "binary_train.csv"
    frame.to_csv(data_path, index=False)

    run_default = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 42},
            "postprocess": {"calibration": "platt"},
            "export": {"artifact_dir": str(tmp_path / "artifacts_default")},
        }
    )
    artifact_default = Artifact.load(run_default.artifact_path)
    assert artifact_default.threshold == {"policy": "fixed", "value": 0.5}
    assert not (Path(run_default.artifact_path) / "threshold_curve.csv").exists()

    run_opt = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 42},
            "postprocess": {
                "calibration": "platt",
                "threshold_optimization": {"enabled": True, "objective": "f1"},
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts_opt")},
        }
    )
    assert run_opt.metadata["threshold_policy"] == "optimized_f1"
    artifact_opt = Artifact.load(run_opt.artifact_path)
    assert artifact_opt.threshold is not None
    assert artifact_opt.threshold.get("policy") == "optimized_f1"
    assert artifact_opt.threshold.get("source") == "oof_p_cal"
    assert 0.01 <= float(artifact_opt.threshold.get("value", 0.0)) <= 0.99
    assert (Path(run_opt.artifact_path) / "threshold_curve.csv").exists()


def test_binary_evaluate_uses_artifact_threshold(tmp_path) -> None:
    frame = _binary_frame(seed=102)
    data_path = tmp_path / "binary_eval.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 42},
            "postprocess": {
                "calibration": "platt",
                "threshold_optimization": {"enabled": True, "objective": "f1"},
            },
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)
    result = evaluate(artifact, frame)
    assert result.metrics["threshold"] == float(artifact.threshold["value"])
    assert result.metadata["threshold_policy"] == artifact.threshold["policy"]
