from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any

from veldra.api.artifact import Artifact
from veldra.artifact import exporter
from veldra.config.models import RunConfig


def _artifact(*, optimize_enabled: bool) -> Artifact:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "export": {
                "artifact_dir": "artifacts",
                "onnx_optimization": {
                    "enabled": optimize_enabled,
                    "mode": "dynamic_quant",
                },
            },
        }
    )
    return Artifact.from_config(
        run_config=cfg,
        run_id="rid_onnx_opt",
        feature_schema={"feature_names": ["x1"], "target": "target"},
        model_text="dummy-model",
    )


def _mock_onnx_toolchain(monkeypatch) -> None:  # type: ignore[no-untyped-def]
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


def test_export_onnx_without_optimization_keeps_default_metadata(monkeypatch, tmp_path) -> None:
    artifact = _artifact(optimize_enabled=False)
    _mock_onnx_toolchain(monkeypatch)
    artifact._get_booster = MethodType(lambda self: object(), artifact)  # type: ignore[method-assign]

    out_dir = tmp_path / "onnx_no_opt"
    exporter.export_onnx_model(artifact, out_dir)
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["onnx_optimized"] is False
    assert metadata["onnx_optimization_mode"] is None
    assert metadata["optimized_model_path"] is None
    assert isinstance(metadata["size_before_bytes"], int)
    assert metadata["size_after_bytes"] is None


def test_export_onnx_with_optimization_writes_optimized_metadata(monkeypatch, tmp_path) -> None:
    artifact = _artifact(optimize_enabled=True)
    _mock_onnx_toolchain(monkeypatch)
    artifact._get_booster = MethodType(lambda self: object(), artifact)  # type: ignore[method-assign]

    def _fake_optimize(model_path: Path, mode: str, out_dir: Path) -> dict[str, Any]:
        optimized = out_dir / "model.optimized.onnx"
        optimized.write_bytes(b"optimized-bytes")
        return {
            "onnx_optimized": True,
            "onnx_optimization_mode": mode,
            "optimized_model_path": str(optimized),
            "size_before_bytes": int(model_path.stat().st_size),
            "size_after_bytes": int(optimized.stat().st_size),
        }

    monkeypatch.setattr(exporter, "_optimize_onnx_model", _fake_optimize)
    out_dir = tmp_path / "onnx_with_opt"
    exporter.export_onnx_model(artifact, out_dir)
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["onnx_optimized"] is True
    assert metadata["onnx_optimization_mode"] == "dynamic_quant"
    assert Path(metadata["optimized_model_path"]).exists()
    assert metadata["size_after_bytes"] is not None
