from pathlib import Path

import numpy as np
import pandas as pd

from veldra.api import Artifact, fit


def _frontier_frame(rows: int = 140, seed: int = 314) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    base = 1.8 + 1.5 * x1 - 0.4 * x2
    noise = rng.normal(scale=0.25 + 0.3 * np.abs(x1), size=rows)
    tail = rng.exponential(scale=0.25, size=rows)
    y = base + noise + tail
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_frontier_fit_smoke_creates_artifact_and_metrics(tmp_path) -> None:
    frame = _frontier_frame()
    data_path = tmp_path / "frontier_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 42},
            "frontier": {"alpha": 0.90},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert run.task_type == "frontier"
    assert {"pinball", "mae", "mean_u_hat", "coverage"} <= set(run.metrics.keys())

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
    assert "frontier_pred" in pred.columns
