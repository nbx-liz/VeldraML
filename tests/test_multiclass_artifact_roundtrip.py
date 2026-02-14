import numpy as np

from veldra.api import Artifact, evaluate, fit


def _train_artifact(tmp_path, multiclass_frame) -> Artifact:
    frame = multiclass_frame(rows_per_class=35, seed=44, scale=0.4)
    data_path = tmp_path / "multiclass_train.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 4},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    return Artifact.load(run.artifact_path)


def test_multiclass_artifact_roundtrip_predict_and_evaluate(tmp_path, multiclass_frame) -> None:
    artifact = _train_artifact(tmp_path, multiclass_frame)
    copied_dir = tmp_path / "copied_artifact"
    artifact.save(copied_dir)
    copied = Artifact.load(copied_dir)

    frame = multiclass_frame(rows_per_class=10, seed=45, scale=0.4)
    x = frame[["x1", "x2"]]
    original_pred = artifact.predict(x)
    copied_pred = copied.predict(x)

    assert list(original_pred.columns) == list(copied_pred.columns)
    prob_cols = [col for col in original_pred.columns if col.startswith("proba_")]
    assert np.allclose(
        original_pred[prob_cols].to_numpy(),
        copied_pred[prob_cols].to_numpy(),
        atol=1e-12,
    )
    assert original_pred["label_pred"].tolist() == copied_pred["label_pred"].tolist()

    original_eval = evaluate(artifact, frame)
    copied_eval = evaluate(copied, frame)
    assert original_eval.metrics.keys() == copied_eval.metrics.keys()
