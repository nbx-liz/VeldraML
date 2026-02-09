import pytest

from veldra.api import (
    Artifact,
    VeldraNotImplementedError,
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
        "data": {"target": "y"},
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
    payload = _config_payload()
    payload["export"] = {"artifact_dir": str(tmp_path)}
    run = fit(payload)
    artifact = Artifact.load(run.artifact_path)

    with pytest.raises(VeldraNotImplementedError):
        tune(payload)
    with pytest.raises(VeldraNotImplementedError):
        evaluate(payload, data=None)
    with pytest.raises(VeldraNotImplementedError):
        predict(artifact, data=None)
    with pytest.raises(VeldraNotImplementedError):
        simulate(artifact, data=None, scenarios=None)
    with pytest.raises(VeldraNotImplementedError):
        export(artifact, format="python")
