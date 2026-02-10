from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import Artifact, export, fit
from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError
from veldra.config.models import RunConfig


def _regression_payload(data_path: str, artifact_dir: str) -> dict:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": data_path, "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 42},
        "export": {"artifact_dir": artifact_dir},
    }


def test_export_runner_returns_contract(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            "x2": [1.0, 0.9, 1.1, 0.2, 0.1, 0.3],
            "target": [0.8, 0.9, 1.0, 1.8, 1.9, 2.1],
        }
    )
    data_path = tmp_path / "reg.csv"
    frame.to_csv(data_path, index=False)
    run = fit(_regression_payload(str(data_path), str(tmp_path)))
    artifact = Artifact.load(run.artifact_path)

    result = export(artifact, format="python")
    assert result.format == "python"
    assert result.path
    assert result.metadata["task_type"] == "regression"
    assert result.metadata["run_id"] == artifact.manifest.run_id
    assert result.metadata["files"]


def test_export_runner_validates_format_and_artifact_model(tmp_path) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    artifact = Artifact.from_config(
        run_config=cfg,
        run_id="rid_export",
        feature_schema={"feature_names": ["x1"], "target": "target"},
        model_text="dummy-model",
    )
    with pytest.raises(VeldraValidationError, match="Supported formats are"):
        export(artifact, format="zip")

    artifact.model_text = None
    with pytest.raises(VeldraValidationError, match="model is missing"):
        export(artifact, format="python")


def test_export_runner_rejects_unknown_task(tmp_path) -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    artifact = Artifact.from_config(
        run_config=cfg,
        run_id="rid_export_unknown",
        feature_schema={"feature_names": ["x1"], "target": "target"},
        model_text="dummy-model",
    )
    artifact.run_config.task.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraNotImplementedError):
        export(artifact, format="python")
