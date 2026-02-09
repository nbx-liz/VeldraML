import pandas as pd
import pytest

from veldra.api import (
    Artifact,
    VeldraNotImplementedError,
    VeldraValidationError,
    evaluate,
    export,
    fit,
    predict,
    simulate,
    tune,
)


def _config_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "", "target": "y"},
    }


def test_api_symbols_are_importable() -> None:
    assert callable(fit)
    assert callable(tune)
    assert callable(evaluate)
    assert callable(predict)
    assert callable(simulate)
    assert callable(export)
    assert hasattr(Artifact, "load")
    assert hasattr(Artifact, "save")
    assert hasattr(Artifact, "predict")
    assert hasattr(Artifact, "simulate")


def test_unimplemented_runner_endpoints_raise_consistent_error(tmp_path) -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "y": [0.2, 1.1, 1.9, 3.2]})
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)

    payload = _config_payload()
    payload["data"]["path"] = str(data_path)
    payload["split"] = {"type": "kfold", "n_splits": 2, "seed": 7}
    payload["export"] = {"artifact_dir": str(tmp_path)}
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    with pytest.raises(VeldraNotImplementedError):
        tune(payload)
    with pytest.raises(VeldraNotImplementedError):
        evaluate(payload, data=None)
    with pytest.raises(VeldraValidationError):
        predict(artifact, data=None)
    pred = predict(artifact, data=frame[["x1"]])
    assert len(pred.data) == len(frame)
    with pytest.raises(VeldraNotImplementedError):
        simulate(artifact, data=None, scenarios=None)
    with pytest.raises(VeldraNotImplementedError):
        export(artifact, format="python")
