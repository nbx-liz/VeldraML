from __future__ import annotations

import builtins
from pathlib import Path
from types import MethodType

import pytest

from veldra.api.artifact import Artifact
from veldra.api.exceptions import VeldraValidationError
from veldra.artifact import exporter
from veldra.config.models import RunConfig


def _artifact() -> Artifact:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "export": {
                "artifact_dir": "artifacts",
                "onnx_optimization": {"enabled": True, "mode": "dynamic_quant"},
            },
        }
    )
    return Artifact.from_config(
        run_config=cfg,
        run_id="rid_onnx_opt_err",
        feature_schema={"feature_names": ["x1"], "target": "target"},
        model_text="dummy-model",
    )


def _mock_export_onnx(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _FakeONNXModel:
        def SerializeToString(self) -> bytes:
            return b"onnx-bytes"

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


def test_optimize_onnx_model_guides_when_dependency_missing(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"raw-onnx")
    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "onnxruntime.quantization":
            raise ModuleNotFoundError(
                "No module named 'onnxruntime.quantization'",
                name="onnxruntime.quantization",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(VeldraValidationError, match="uv sync --extra export-onnx"):
        exporter._optimize_onnx_model(
            model_path=model_path,
            mode="dynamic_quant",
            out_dir=tmp_path,
        )


def test_export_onnx_model_raises_on_quantization_failure(monkeypatch, tmp_path) -> None:
    artifact = _artifact()
    _mock_export_onnx(monkeypatch)
    artifact._get_booster = MethodType(lambda self: object(), artifact)  # type: ignore[method-assign]

    def _boom(model_path: Path, mode: str, out_dir: Path) -> dict[str, object]:
        _ = model_path, mode, out_dir
        raise VeldraValidationError("ONNX dynamic quantization failed")

    monkeypatch.setattr(exporter, "_optimize_onnx_model", _boom)
    with pytest.raises(VeldraValidationError, match="ONNX dynamic quantization failed"):
        exporter.export_onnx_model(artifact, tmp_path / "onnx_quant_fail")
