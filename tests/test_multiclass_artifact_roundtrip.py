import numpy as np
import pandas as pd

from veldra.api import Artifact, evaluate, fit


def _multiclass_frame(rows_per_class: int = 35, seed: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["alpha", "beta", "gamma"]
    frames: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        center = float(idx) * 1.8
        x1 = rng.normal(loc=center, scale=0.4, size=rows_per_class)
        x2 = rng.normal(loc=2.0 - center, scale=0.4, size=rows_per_class)
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
            "split": {"type": "stratified", "n_splits": 3, "seed": 4},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    return Artifact.load(run.artifact_path)


def test_multiclass_artifact_roundtrip_predict_and_evaluate(tmp_path) -> None:
    artifact = _train_artifact(tmp_path)
    copied_dir = tmp_path / "copied_artifact"
    artifact.save(copied_dir)
    copied = Artifact.load(copied_dir)

    frame = _multiclass_frame(rows_per_class=10, seed=45)
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
