from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import Artifact, fit


def _binary_frame(rows: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 1.8 * x1 - 1.2 * x2 + rng.normal(scale=0.4, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_binary_fit_smoke_creates_artifact_with_calibration_files(tmp_path) -> None:
    frame = _binary_frame()
    data_path = tmp_path / "binary_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 42},
            "postprocess": {"calibration": "platt", "threshold": 0.5},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert run.task_type == "binary"
    assert {"auc", "logloss", "brier"} <= set(run.metrics.keys())

    artifact_path = Path(run.artifact_path)
    assert artifact_path.exists()
    expected_files = {
        "manifest.json",
        "run_config.yaml",
        "feature_schema.json",
        "model.lgb.txt",
        "metrics.json",
        "cv_results.parquet",
        "calibrator.pkl",
        "calibration_curve.csv",
        "threshold.json",
    }
    assert expected_files <= {path.name for path in artifact_path.iterdir()}

    artifact = Artifact.load(artifact_path)
    pred = artifact.predict(frame[["x1", "x2"]])
    assert list(pred.columns) == ["p_cal", "p_raw", "label_pred"]
