import numpy as np
import pandas as pd

from veldra.api import Artifact, fit, predict


def test_regression_predict_roundtrip(tmp_path) -> None:
    train_df = pd.DataFrame(
        {
            "x1": [float(i) for i in range(30)],
            "x2": [float(i % 5) for i in range(30)],
            "y": [float(i * 2.0 + (i % 5) * 0.3) for i in range(30)],
        }
    )
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(train_path), "target": "y"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 11},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
            "train": {"lgb_params": {"num_threads": 1}, "seed": 11},
        }
    )

    artifact = Artifact.load(run.artifact_path)
    infer_df = train_df[["x1", "x2"]].copy()
    pred = predict(artifact, infer_df)

    assert pred.task_type == "regression"
    assert pred.metadata["n_rows"] == len(infer_df)
    assert isinstance(pred.data, np.ndarray)
    assert pred.data.shape[0] == len(infer_df)
