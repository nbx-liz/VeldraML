from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import Artifact, fit


def _multiclass_frame(rows_per_class: int = 40, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["alpha", "beta", "gamma"]
    frames: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        center = float(idx) * 2.5
        x1 = rng.normal(loc=center, scale=0.35, size=rows_per_class)
        x2 = rng.normal(loc=-center, scale=0.35, size=rows_per_class)
        frames.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(frames, ignore_index=True)


def test_multiclass_fit_smoke_creates_artifact_and_metrics(tmp_path) -> None:
    frame = _multiclass_frame()
    data_path = tmp_path / "multiclass_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 42},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert run.task_type == "multiclass"
    assert {"accuracy", "macro_f1", "logloss"} <= set(run.metrics.keys())

    artifact_path = Path(run.artifact_path)
    assert artifact_path.exists()
    expected_files = {
        "manifest.json",
        "run_config.yaml",
        "feature_schema.json",
        "model.lgb.txt",
        "metrics.json",
        "cv_results.parquet",
    }
    assert expected_files <= {path.name for path in artifact_path.iterdir()}

    artifact = Artifact.load(artifact_path)
    pred = artifact.predict(frame[["x1", "x2"]])
    assert "label_pred" in pred.columns
    assert {"proba_alpha", "proba_beta", "proba_gamma"} <= set(pred.columns)
