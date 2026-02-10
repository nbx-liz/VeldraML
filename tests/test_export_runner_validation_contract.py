from __future__ import annotations

from pathlib import Path

import pandas as pd

from veldra.api import Artifact, export, fit
from veldra.api import runner as runner_module


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


def test_export_runner_metadata_contains_validation_contract(tmp_path) -> None:
    artifact = _fit_regression_artifact(tmp_path)
    result = export(artifact, format="python")

    assert result.metadata["validation_mode"] == "python"
    assert isinstance(result.metadata["validation_passed"], bool)
    report_path = Path(result.metadata["validation_report"])
    assert report_path.exists()
    assert "validation_report.json" in result.metadata["files"]
    assert "onnx_optimized" in result.metadata
    assert "onnx_optimization_mode" in result.metadata
    assert "optimized_model_path" in result.metadata
    assert "size_before_bytes" in result.metadata
    assert "size_after_bytes" in result.metadata


def test_export_runner_propagates_validation_failure_status(monkeypatch, tmp_path) -> None:
    artifact = _fit_regression_artifact(tmp_path)

    def _fake_validate_python_export(out_dir, local_artifact):  # type: ignore[no-untyped-def]
        _ = local_artifact
        report = Path(out_dir) / "validation_report.json"
        report.write_text("{}", encoding="utf-8")
        return {
            "validation_mode": "python",
            "validation_passed": False,
            "validation_report": str(report),
        }

    monkeypatch.setattr(runner_module, "_validate_python_export", _fake_validate_python_export)
    result = export(artifact, format="python")
    assert result.metadata["validation_mode"] == "python"
    assert result.metadata["validation_passed"] is False
