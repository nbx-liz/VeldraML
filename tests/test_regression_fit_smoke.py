from pathlib import Path

import pandas as pd

from veldra.api import fit


def test_regression_fit_smoke_outputs_artifact_files(tmp_path: Path) -> None:
    train_df = pd.DataFrame(
        {
            "x1": [float(i) for i in range(20)],
            "x2": [float(i % 3) for i in range(20)],
            "y": [float(i * 1.5 + 0.2) for i in range(20)],
        }
    )
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False)

    result = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(train_path), "target": "y"},
            "split": {"type": "kfold", "n_splits": 4, "seed": 13},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
            "train": {"lgb_params": {"num_threads": 1}, "seed": 13},
        }
    )

    artifact_path = Path(result.artifact_path or "")
    assert artifact_path.exists()
    assert (artifact_path / "manifest.json").exists()
    assert (artifact_path / "run_config.yaml").exists()
    assert (artifact_path / "feature_schema.json").exists()
    assert (artifact_path / "model.lgb.txt").exists()
    assert (artifact_path / "metrics.json").exists()
    assert (artifact_path / "cv_results.parquet").exists()
    assert result.metrics
