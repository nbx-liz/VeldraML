from __future__ import annotations

import builtins
import json
from types import MethodType
from typing import Any

import pytest

from veldra.api.artifact import Artifact
from veldra.api.exceptions import VeldraValidationError
from veldra.artifact import exporter
from veldra.config.models import RunConfig


def _config(task_type: str = "regression") -> RunConfig:
    payload: dict[str, Any] = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": "dummy.csv", "target": "target"},
        "export": {"artifact_dir": "artifacts"},
    }
    if task_type == "frontier":
        payload["frontier"] = {"alpha": 0.90}
    return RunConfig.model_validate(payload)


def _artifact(
    *,
    task_type: str = "regression",
    model_text: str | None = "dummy-model",
    feature_schema: dict[str, Any] | None = None,
) -> Artifact:
    schema = {"feature_names": ["x1"], "target": "target"} if feature_schema is None else feature_schema
    return Artifact.from_config(
        run_config=_config(task_type),
        run_id="rid_exporter_test",
        feature_schema=schema,
        model_text=model_text,
    )


def test_export_python_package_validates_required_fields(tmp_path) -> None:
    artifact_missing_model = _artifact(model_text=None)
    with pytest.raises(VeldraValidationError, match="model is missing"):
        exporter.export_python_package(artifact_missing_model, tmp_path / "python_missing_model")

    artifact_missing_schema = _artifact(feature_schema={})
    with pytest.raises(VeldraValidationError, match="feature_schema is missing"):
        exporter.export_python_package(artifact_missing_schema, tmp_path / "python_missing_schema")


def test_load_onnx_toolchain_missing_dependency_returns_guidance(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "onnxmltools":
            raise ModuleNotFoundError("No module named 'onnxmltools'", name="onnxmltools")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(VeldraValidationError, match="uv sync --extra export-onnx"):
        exporter._load_onnx_toolchain()


def test_export_onnx_model_validates_required_inputs(tmp_path) -> None:
    artifact_missing_model = _artifact(model_text=None)
    with pytest.raises(VeldraValidationError, match="model is missing"):
        exporter.export_onnx_model(artifact_missing_model, tmp_path / "onnx_missing_model")

    artifact_missing_feature_names = _artifact(feature_schema={"target": "target"})
    with pytest.raises(VeldraValidationError, match="feature_schema.feature_names"):
        exporter.export_onnx_model(artifact_missing_feature_names, tmp_path / "onnx_missing_schema")


def test_export_onnx_model_writes_outputs_with_mocked_toolchain(monkeypatch, tmp_path) -> None:
    artifact = _artifact()

    class _FakeONNXModel:
        def SerializeToString(self) -> bytes:
            return b"onnx-bytes"

    class _FakeOnnxTools:
        def __init__(self) -> None:
            self.last_initial_types = None

        def convert_lightgbm(self, booster, initial_types):  # type: ignore[no-untyped-def]
            _ = booster
            self.last_initial_types = initial_types
            return _FakeONNXModel()

    class _FakeFloatTensorType:
        def __init__(self, shape):  # type: ignore[no-untyped-def]
            self.shape = shape

    fake_toolchain = _FakeOnnxTools()
    monkeypatch.setattr(
        exporter,
        "_load_onnx_toolchain",
        lambda: (fake_toolchain, _FakeFloatTensorType),
    )

    artifact._get_booster = MethodType(lambda self: object(), artifact)  # type: ignore[method-assign]
    out_dir = tmp_path / "onnx_export"
    result_dir = exporter.export_onnx_model(artifact, out_dir)

    assert result_dir == out_dir
    model_path = out_dir / "model.onnx"
    metadata_path = out_dir / "metadata.json"
    assert model_path.exists()
    assert metadata_path.exists()
    assert model_path.read_bytes() == b"onnx-bytes"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == artifact.manifest.run_id
    assert metadata["task_type"] == "regression"
    assert metadata["export_format"] == "onnx"
    assert fake_toolchain.last_initial_types is not None
