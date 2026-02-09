import numpy as np
import pandas as pd

from veldra.api import Artifact, evaluate, fit


def _binary_frame(rows: int = 100, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 0.8 * x1 - 1.5 * x2 + rng.normal(scale=0.35, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_binary_artifact_save_load_roundtrip(tmp_path) -> None:
    frame = _binary_frame()
    data_path = tmp_path / "binary_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 8},
            "postprocess": {"calibration": "platt", "threshold": 0.5},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    loaded = Artifact.load(run.artifact_path)

    copied_path = tmp_path / "copied_artifact"
    loaded.save(copied_path)
    copied = Artifact.load(copied_path)

    pred = copied.predict(frame[["x1", "x2"]])
    assert list(pred.columns) == ["p_cal", "p_raw", "label_pred"]
    assert copied.calibrator is not None
    assert copied.threshold == {"policy": "fixed", "value": 0.5}

    eval_result = evaluate(copied, frame)
    assert {"auc", "logloss", "brier"} <= set(eval_result.metrics)
