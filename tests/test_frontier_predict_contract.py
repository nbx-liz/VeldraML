import numpy as np

from veldra.api import Artifact, fit, predict


def test_frontier_predict_contract(tmp_path, frontier_frame) -> None:
    frame = frontier_frame(rows=120, seed=500)
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
