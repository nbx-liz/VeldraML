from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.api.artifact import Artifact
from veldra.api.exceptions import (
    VeldraArtifactError,
    VeldraNotImplementedError,
    VeldraValidationError,
)
from veldra.config.models import RunConfig


def _artifact(task_type: str, feature_schema: dict | None = None) -> Artifact:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": task_type},
            "data": {"path": "dummy.csv", "target": "target"},
        }
    )
    return Artifact.from_config(
        run_config=cfg,
        run_id=f"rid_{task_type}",
        feature_schema=feature_schema or {"feature_names": ["x1"]},
    )


def test_get_booster_requires_model_text() -> None:
    artifact = _artifact("regression")
    with pytest.raises(VeldraValidationError, match="model is missing"):
        artifact._get_booster()


def test_prepare_feature_frame_validation_errors() -> None:
    artifact = _artifact("regression")
    artifact.feature_schema = {}
    with pytest.raises(VeldraValidationError, match="feature_schema.feature_names"):
        artifact._prepare_feature_frame(pd.DataFrame({"x1": [1.0]}))

    artifact.feature_schema = {"feature_names": ["x1", "x2"]}
    with pytest.raises(VeldraValidationError, match="missing required feature columns"):
        artifact._prepare_feature_frame(pd.DataFrame({"x1": [1.0]}))


def test_predict_binary_requires_calibrator(monkeypatch) -> None:
    artifact = _artifact("binary")

    class _Booster:
        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return np.full(len(x), 0.5)

    monkeypatch.setattr(artifact, "_get_booster", lambda: _Booster())
    with pytest.raises(VeldraArtifactError, match="missing calibrator"):
        artifact._predict_binary(pd.DataFrame({"x1": [1.0, 2.0]}))


def test_predict_multiclass_validation_errors(monkeypatch) -> None:
    artifact = _artifact("multiclass", feature_schema={"feature_names": ["x1"]})

    class _BoosterBad:
        def __init__(self, out: np.ndarray) -> None:
            self.out = out

        def predict(self, _: pd.DataFrame) -> np.ndarray:
            return self.out

    with pytest.raises(VeldraArtifactError, match="target_classes"):
        artifact._predict_multiclass(pd.DataFrame({"x1": [1.0, 2.0]}))

    artifact.feature_schema["target_classes"] = ["a", "b", "c"]
    monkeypatch.setattr(
        artifact,
        "_get_booster",
        lambda: _BoosterBad(np.array([0.2, 0.3, 0.5, 0.1])),
    )
    with pytest.raises(VeldraArtifactError, match="invalid shape"):
        artifact._predict_multiclass(pd.DataFrame({"x1": [1.0, 2.0]}))

    monkeypatch.setattr(
        artifact,
        "_get_booster",
        lambda: _BoosterBad(np.array([[0.2, 0.8], [0.4, 0.6]])),
    )
    with pytest.raises(VeldraArtifactError, match="invalid dimensions"):
        artifact._predict_multiclass(pd.DataFrame({"x1": [1.0, 2.0]}))


def test_predict_and_simulate_unimplemented_paths() -> None:
    artifact = _artifact("frontier")
    with pytest.raises(VeldraValidationError, match="pandas.DataFrame"):
        artifact.predict([1, 2, 3])
    with pytest.raises(VeldraValidationError, match="model is missing"):
        artifact.predict(pd.DataFrame({"x1": [1.0]}))
    artifact.run_config.task.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraNotImplementedError, match="implemented only"):
        artifact.predict(pd.DataFrame({"x1": [1.0]}))
    with pytest.raises(VeldraNotImplementedError, match="not implemented"):
        artifact.simulate(pd.DataFrame({"x1": [1.0]}), scenario={})


def test_predict_multiclass_accepts_flat_output_with_expected_size(monkeypatch) -> None:
    artifact = _artifact(
        "multiclass",
        feature_schema={"feature_names": ["x1"], "target_classes": ["a", "b", "c"]},
    )

    class _BoosterFlat:
        def predict(self, _: pd.DataFrame) -> np.ndarray:
            return np.array([0.2, 0.3, 0.5, 0.5, 0.2, 0.3], dtype=float)

    monkeypatch.setattr(artifact, "_get_booster", lambda: _BoosterFlat())
    pred = artifact._predict_multiclass(pd.DataFrame({"x1": [1.0, 2.0]}))

    assert list(pred.columns) == ["label_pred", "proba_a", "proba_b", "proba_c"]
    assert len(pred) == 2
