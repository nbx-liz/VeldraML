import numpy as np

from veldra.api import Artifact, evaluate, fit


def test_regression_artifact_save_load_roundtrip(tmp_path, regression_frame) -> None:
    frame = regression_frame(rows=100, seed=5)
    data_path = tmp_path / "regression_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    loaded = Artifact.load(run.artifact_path)

    copied_path = tmp_path / "copied_artifact"
    loaded.save(copied_path)
    copied = Artifact.load(copied_path)

    x = frame[["x1", "x2"]]
    pred_original = loaded.predict(x)
    pred_copied = copied.predict(x)
    assert np.allclose(pred_original, pred_copied, atol=1e-12)
    assert copied.feature_schema == loaded.feature_schema

    eval_result = evaluate(copied, frame)
    assert {"rmse", "mae", "r2"} <= set(eval_result.metrics)
