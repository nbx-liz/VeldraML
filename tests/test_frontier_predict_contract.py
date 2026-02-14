import numpy as np
import pandas as pd

from veldra.api import Artifact, fit, predict


def _frontier_frame(rows: int = 120, seed: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.5, 2.5, size=rows)
    x2 = rng.normal(size=rows)
    y = (
        2.0
        + 1.2 * x1
        - 0.5 * x2
        + rng.normal(scale=0.3, size=rows)
        + rng.exponential(scale=0.2, size=rows)
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_frontier_predict_contract(tmp_path) -> None:
    frame = _frontier_frame()
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 9},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)

    pred_unlabeled = predict(artifact, frame[["x1", "x2"]]).data
    assert list(pred_unlabeled.columns) == ["frontier_pred"]
    assert len(pred_unlabeled) == len(frame)

    pred_labeled = artifact.predict(frame[["x1", "x2", "target"]])
    assert {"frontier_pred", "u_hat"} <= set(pred_labeled.columns)
    assert np.all(pred_labeled["u_hat"].to_numpy() >= 0.0)
