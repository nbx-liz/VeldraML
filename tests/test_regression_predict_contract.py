import numpy as np
import pytest

from veldra.api import Artifact, fit, predict
from veldra.api.exceptions import VeldraValidationError


def test_regression_predict_contract(tmp_path, regression_frame) -> None:
    frame = regression_frame(rows=90, seed=31)
    data_path = tmp_path / "regression.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 4},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)

    pred_from_runner = predict(artifact, frame[["x1", "x2"]]).data
    pred_from_artifact = artifact.predict(frame[["x1", "x2"]])
    assert isinstance(pred_from_runner, np.ndarray)
    assert isinstance(pred_from_artifact, np.ndarray)
    assert len(pred_from_runner) == len(frame)
    assert len(pred_from_artifact) == len(frame)

    reversed_order = artifact.predict(frame[["x2", "x1"]])
    assert len(reversed_order) == len(frame)

    with pytest.raises(VeldraValidationError, match="missing required feature columns"):
        artifact.predict(frame[["x1"]])
