from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import MethodType, ModuleType
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
    schema = (
        {"feature_names": ["x1"], "target": "target"} if feature_schema is None else feature_schema
    )
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


def test_export_python_package_writes_expected_files_and_metadata(tmp_path) -> None:
    artifact = _artifact()
    out_dir = tmp_path / "python_export"
    result_dir = exporter.export_python_package(artifact, out_dir)

    assert result_dir == out_dir
    expected_files = {
        "manifest.json",
        "run_config.yaml",
        "feature_schema.json",
        "model.lgb.txt",
        "metadata.json",
        "runtime_predict.py",
        "README.md",
    }
    assert expected_files.issubset({p.name for p in out_dir.iterdir()})

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["task_type"] == "regression"
    assert metadata["export_format"] == "python"
    assert metadata["run_id"] == "rid_exporter_test"


def test_metadata_payload_frontier_uses_config_alpha_fallback() -> None:
    artifact = _artifact(
        task_type="frontier",
        feature_schema={"feature_names": ["x1"], "target": "target"},
    )
    payload = exporter._metadata_payload(artifact, export_format="onnx")
    assert payload["task_type"] == "frontier"
    assert payload["frontier_alpha"] == pytest.approx(artifact.run_config.frontier.alpha)


def test_load_onnx_toolchain_missing_dependency_returns_guidance(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "onnxmltools":
            raise ModuleNotFoundError("No module named 'onnxmltools'", name="onnxmltools")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(VeldraValidationError, match="uv sync --extra export-onnx"):
        exporter._load_onnx_toolchain()


def test_load_onnx_toolchain_success_when_optional_deps_installed() -> None:
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnxconverter_common")
    onnx_tools, tensor_type = exporter._load_onnx_toolchain()
    assert onnx_tools is not None
    assert tensor_type is not None


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


def test_export_onnx_model_writes_frontier_metadata_with_mocked_toolchain(
    monkeypatch, tmp_path
) -> None:
    artifact = _artifact(
        task_type="frontier",
        feature_schema={"feature_names": ["x1"], "target": "target", "frontier_alpha": 0.9},
    )

    class _FakeONNXModel:
        def SerializeToString(self) -> bytes:
            return b"frontier-onnx"

    class _FakeOnnxTools:
        def convert_lightgbm(self, booster, initial_types):  # type: ignore[no-untyped-def]
            _ = booster, initial_types
            return _FakeONNXModel()

    class _FakeFloatTensorType:
        def __init__(self, shape):  # type: ignore[no-untyped-def]
            self.shape = shape

    monkeypatch.setattr(
        exporter,
        "_load_onnx_toolchain",
        lambda: (_FakeOnnxTools(), _FakeFloatTensorType),
    )
    artifact._get_booster = MethodType(lambda self: object(), artifact)  # type: ignore[method-assign]

    out_dir = tmp_path / "onnx_export_frontier"
    exporter.export_onnx_model(artifact, out_dir)

    metadata_path = out_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["task_type"] == "frontier"
    assert payload["frontier_alpha"] == 0.9


def test_export_onnx_model_converter_failure_has_actionable_message(monkeypatch, tmp_path) -> None:
    artifact = _artifact(task_type="frontier")

    class _FailingOnnxTools:
        def convert_lightgbm(self, booster, initial_types):  # type: ignore[no-untyped-def]
            _ = booster, initial_types
            raise RuntimeError("converter boom")

    class _FakeFloatTensorType:
        def __init__(self, shape):  # type: ignore[no-untyped-def]
            self.shape = shape

    monkeypatch.setattr(
        exporter,
        "_load_onnx_toolchain",
        lambda: (_FailingOnnxTools(), _FakeFloatTensorType),
    )
    artifact._get_booster = MethodType(lambda self: object(), artifact)  # type: ignore[method-assign]

    with pytest.raises(VeldraValidationError, match="ONNX conversion failed"):
        exporter.export_onnx_model(artifact, tmp_path / "onnx_export_fail")


def test_export_onnx_model_write_failure_has_actionable_message(monkeypatch, tmp_path) -> None:
    artifact = _artifact()

    class _FakeONNXModel:
        def SerializeToString(self) -> bytes:
            raise OSError("serialize failure")

    class _FakeOnnxTools:
        def convert_lightgbm(self, booster, initial_types):  # type: ignore[no-untyped-def]
            _ = booster, initial_types
            return _FakeONNXModel()

    class _FakeFloatTensorType:
        def __init__(self, shape):  # type: ignore[no-untyped-def]
            self.shape = shape

    monkeypatch.setattr(
        exporter,
        "_load_onnx_toolchain",
        lambda: (_FakeOnnxTools(), _FakeFloatTensorType),
    )
    artifact._get_booster = MethodType(lambda self: object(), artifact)  # type: ignore[method-assign]

    with pytest.raises(
        VeldraValidationError,
        match="Failed to serialize/write ONNX model artifact",
    ):
        exporter.export_onnx_model(artifact, tmp_path / "onnx_export_write_fail")


def test_optimize_onnx_model_rejects_unknown_mode(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx-bytes")

    with pytest.raises(VeldraValidationError, match="Unsupported ONNX optimization mode"):
        exporter._optimize_onnx_model(
            model_path=model_path,
            mode="static_quant",
            out_dir=tmp_path,
        )


def test_optimize_onnx_model_success_with_mocked_quantizer(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"raw-onnx-bytes")

    quant_mod = ModuleType("onnxruntime.quantization")

    class _QuantType:
        QInt8 = "QInt8"

    def _quantize_dynamic(*, model_input: str, model_output: str, weight_type: Any) -> None:
        _ = model_input, weight_type
        Path(model_output).write_bytes(b"optimized-onnx")

    quant_mod.QuantType = _QuantType
    quant_mod.quantize_dynamic = _quantize_dynamic
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quant_mod)

    result = exporter._optimize_onnx_model(
        model_path=model_path,
        mode="dynamic_quant",
        out_dir=tmp_path,
    )
    assert result["onnx_optimized"] is True
    assert result["onnx_optimization_mode"] == "dynamic_quant"
    assert Path(result["optimized_model_path"]).exists()
    assert result["size_before_bytes"] == len(b"raw-onnx-bytes")
    assert result["size_after_bytes"] == len(b"optimized-onnx")


def test_validate_python_export_marks_invalid_feature_names_when_runtime_exists(tmp_path) -> None:
    artifact = _artifact()
    export_dir = exporter.export_python_package(
        artifact,
        tmp_path / "python_validate_invalid_schema",
    )
    artifact.feature_schema = {"target": "target"}  # runtime file exists, but schema is invalid

    report = exporter._validate_python_export(export_dir, artifact)
    payload = json.loads(Path(report["validation_report"]).read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["runtime_predict"]["ok"] is False
    assert (
        "feature_schema.feature_names is missing or invalid" in checks["runtime_predict"]["detail"]
    )


def test_validate_python_export_records_subprocess_exception(monkeypatch, tmp_path) -> None:
    artifact = _artifact()
    export_dir = exporter.export_python_package(
        artifact,
        tmp_path / "python_validate_subprocess_error",
    )

    def _raise(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise RuntimeError("probe boom")

    monkeypatch.setattr(exporter.subprocess, "run", _raise)
    report = exporter._validate_python_export(export_dir, artifact)
    payload = json.loads(Path(report["validation_report"]).read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["runtime_predict"]["ok"] is False
    assert "probe boom" in checks["runtime_predict"]["detail"]


def test_validate_onnx_export_records_metadata_parse_failure(monkeypatch, tmp_path) -> None:
    artifact = _artifact()
    export_dir = tmp_path / "onnx_bad_metadata"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model.onnx").write_bytes(b"onnx-bytes")
    (export_dir / "metadata.json").write_text("{bad_json", encoding="utf-8")

    original_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    report = exporter._validate_onnx_export(export_dir, artifact)
    payload = json.loads(Path(report["validation_report"]).read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["metadata_parse"]["ok"] is False
    assert checks["metadata_parse"]["detail"] == "Failed to parse metadata.json"


def test_validate_onnx_export_handles_missing_feature_names_with_mocked_runtime(
    monkeypatch, tmp_path
) -> None:
    artifact = _artifact(feature_schema={"target": "target"})
    export_dir = tmp_path / "onnx_missing_feature_names"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model.onnx").write_bytes(b"onnx-bytes")
    (export_dir / "metadata.json").write_text("{}", encoding="utf-8")

    fake_onnx = ModuleType("onnx")

    class _FakeChecker:
        @staticmethod
        def check_model(model: Any) -> None:
            _ = model

    fake_onnx.load = lambda path: object()  # type: ignore[assignment]
    fake_onnx.checker = _FakeChecker()

    fake_ort = ModuleType("onnxruntime")

    class _FakeSession:
        def __init__(self, path: str, providers: list[str]) -> None:
            _ = path, providers

    fake_ort.InferenceSession = _FakeSession  # type: ignore[assignment]

    original_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnx":
            return fake_onnx
        if name == "onnxruntime":
            return fake_ort
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    report = exporter._validate_onnx_export(export_dir, artifact)
    payload = json.loads(Path(report["validation_report"]).read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["onnx_model_check"]["ok"] is True
    assert checks["onnx_runtime_inference"]["ok"] is False
    assert (
        "feature_schema.feature_names is missing or invalid"
        in checks["onnx_runtime_inference"]["detail"]
    )
