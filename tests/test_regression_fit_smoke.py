from pathlib import Path

from veldra.api import Artifact, fit


def test_regression_fit_smoke_creates_artifact_and_metrics(tmp_path, regression_frame) -> None:
    frame = regression_frame(rows=120, seed=11)
    data_path = tmp_path / "regression_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 42},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    assert run.task_type == "regression"
    assert {"rmse", "mae", "r2"} <= set(run.metrics.keys())

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
    assert len(pred) == len(frame)
