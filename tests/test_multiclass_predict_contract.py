import numpy as np

from veldra.api import Artifact, fit, predict


def _train_artifact(tmp_path, multiclass_frame) -> Artifact:
    frame = multiclass_frame(rows_per_class=35, seed=19, scale=0.45)
    data_path = tmp_path / "multiclass_train.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 7},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    return Artifact.load(run.artifact_path)


def test_multiclass_predict_contract_has_label_and_probabilities(
    tmp_path, multiclass_frame
) -> None:
    artifact = _train_artifact(tmp_path, multiclass_frame)
    frame = multiclass_frame(rows_per_class=8, seed=21, scale=0.45)
    out = predict(artifact, frame[["x1", "x2"]]).data

    assert "label_pred" in out.columns
    prob_cols = [col for col in out.columns if col.startswith("proba_")]
    assert len(prob_cols) == 3
    row_sum = out[prob_cols].sum(axis=1).to_numpy(dtype=float)
    assert np.allclose(row_sum, 1.0, atol=1e-6)
    assert out["label_pred"].isin(["alpha", "beta", "gamma"]).all()
