from veldra.api import Artifact, evaluate, fit, predict


def test_frontier_artifact_roundtrip_predict_and_evaluate(tmp_path, frontier_frame) -> None:
    frame = frontier_frame(rows=100, seed=91)
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
