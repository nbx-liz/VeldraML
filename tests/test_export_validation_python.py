from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from veldra.api import Artifact, export, fit
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


def test_export_python_generates_validation_report(tmp_path) -> None:
    artifact = _fit_regression_artifact(tmp_path)
    result = export(artifact, format="python")

    assert result.metadata["validation_mode"] == "python"
    assert result.metadata["validation_passed"] is True
    report_path = Path(result.metadata["validation_report"])
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["validation_mode"] == "python"
    assert payload["validation_passed"] is True


def test_validate_python_export_detects_missing_required_file(tmp_path) -> None:
    artifact = _fit_regression_artifact(tmp_path)
    export_dir = tmp_path / "python_manual"
    exporter.export_python_package(artifact, export_dir)
    (export_dir / "model.lgb.txt").unlink()

    report = exporter._validate_python_export(export_dir, artifact)
    assert report["validation_mode"] == "python"
    assert report["validation_passed"] is False

    payload = json.loads(Path(report["validation_report"]).read_text(encoding="utf-8"))
    status = {item["name"]: bool(item["ok"]) for item in payload["checks"]}
    assert status["required_file:model.lgb.txt"] is False
