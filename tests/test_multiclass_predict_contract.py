import numpy as np
import pandas as pd

from veldra.api import Artifact, fit, predict


def _multiclass_frame(rows_per_class: int = 35, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["alpha", "beta", "gamma"]
    frames: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        center = float(idx) * 2.2
        x1 = rng.normal(loc=center, scale=0.45, size=rows_per_class)
        x2 = rng.normal(loc=1.5 - center, scale=0.45, size=rows_per_class)
        frames.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(frames, ignore_index=True)


def _train_artifact(tmp_path) -> Artifact:
    frame = _multiclass_frame()
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


def test_multiclass_predict_contract_has_label_and_probabilities(tmp_path) -> None:
    artifact = _train_artifact(tmp_path)
    frame = _multiclass_frame(rows_per_class=8, seed=21)
    out = predict(artifact, frame[["x1", "x2"]]).data

    assert "label_pred" in out.columns
    prob_cols = [col for col in out.columns if col.startswith("proba_")]
    assert len(prob_cols) == 3
    row_sum = out[prob_cols].sum(axis=1).to_numpy(dtype=float)
    assert np.allclose(row_sum, 1.0, atol=1e-6)
    assert out["label_pred"].isin(["alpha", "beta", "gamma"]).all()
