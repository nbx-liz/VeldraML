import numpy as np
import pandas as pd

from veldra.api import Artifact, evaluate, fit, predict


def _frontier_frame(rows: int = 100, seed: int = 91) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    y = (
        2.5
        + 1.1 * x1
        - 0.7 * x2
        + rng.normal(scale=0.35, size=rows)
        + rng.exponential(scale=0.2, size=rows)
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_frontier_artifact_roundtrip_predict_and_evaluate(tmp_path) -> None:
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
    loaded = Artifact.load(run.artifact_path)

    pred = predict(loaded, frame[["x1", "x2"]]).data
    assert "frontier_pred" in pred.columns
    assert len(pred) == len(frame)

    eval_result = evaluate(loaded, frame)
    assert {"pinball", "mae", "mean_u_hat", "coverage"} <= set(eval_result.metrics.keys())
