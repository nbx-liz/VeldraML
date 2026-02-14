from veldra.api import Artifact, fit, predict


def test_binary_predict_returns_expected_columns_and_ranges(tmp_path, binary_frame) -> None:
    frame = binary_frame(rows=90, seed=3, coef1=1.1, coef2=-0.9, noise=0.3)
    data_path = tmp_path / "binary.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 4},
            "postprocess": {"calibration": "platt"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)

    pred_from_runner = predict(artifact, frame[["x1", "x2"]]).data
    pred_from_artifact = artifact.predict(frame[["x1", "x2"]])

    for pred in (pred_from_runner, pred_from_artifact):
        assert list(pred.columns) == ["p_cal", "p_raw", "label_pred"]
        assert len(pred) == len(frame)
        assert pred["p_cal"].between(0, 1).all()
        assert pred["p_raw"].between(0, 1).all()
        assert set(pred["label_pred"].unique().tolist()) <= {0, 1}
