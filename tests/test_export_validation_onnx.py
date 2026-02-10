from __future__ import annotations

import builtins
import json
from pathlib import Path

import pandas as pd
import pytest

from veldra.api import Artifact, fit
from veldra.artifact import exporter


def _fit_regression_artifact(tmp_path: Path) -> Artifact:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            "x2": [1.0, 0.9, 1.1, 0.2, 0.1, 0.3],
            "target": [0.8, 0.9, 1.0, 1.8, 1.9, 2.1],
        }
    )
    data_path = tmp_path / "reg.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 42},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    return Artifact.load(run.artifact_path)


def test_validate_onnx_export_passes_when_dependencies_are_available(tmp_path) -> None:
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    artifact = _fit_regression_artifact(tmp_path)
    export_dir = tmp_path / "onnx_manual"
    exporter.export_onnx_model(artifact, export_dir)

    report = exporter._validate_onnx_export(export_dir, artifact)
    assert report["validation_mode"] == "onnx"
    assert report["validation_passed"] is True
    assert Path(report["validation_report"]).exists()


def test_validate_onnx_export_reports_missing_runtime_dependency(monkeypatch, tmp_path) -> None:
    artifact = _fit_regression_artifact(tmp_path)
    export_dir = tmp_path / "onnx_missing_dep"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model.onnx").write_bytes(b"dummy")
    (export_dir / "metadata.json").write_text("{}", encoding="utf-8")

    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "onnxruntime":
            raise ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    report = exporter._validate_onnx_export(export_dir, artifact)
    assert report["validation_mode"] == "onnx"
    assert report["validation_passed"] is False

    payload = json.loads(Path(report["validation_report"]).read_text(encoding="utf-8"))
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["onnx_runtime_import"]["ok"] is False
    assert "uv sync --extra export-onnx" in checks["onnx_runtime_import"]["detail"]
