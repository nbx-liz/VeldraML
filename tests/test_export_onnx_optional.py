from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veldra.api import Artifact, export, fit
from veldra.api.exceptions import VeldraValidationError
from veldra.artifact import exporter


def _regression_frame(rows: int = 32, seed: int = 1301) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.5 * x1 - 0.9 * x2 + rng.normal(scale=0.2, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _fit_regression(tmp_path) -> Artifact:
    frame = _regression_frame()
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


def _fit_frontier(tmp_path) -> Artifact:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
            "x2": [1.0, 0.9, 1.1, 0.2, 0.1, 0.3],
            "target": [0.8, 0.9, 1.0, 1.8, 1.9, 2.1],
        }
    )
    data_path = tmp_path / "frontier.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 42},
            "frontier": {"alpha": 0.90},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    return Artifact.load(run.artifact_path)


def test_export_onnx_missing_dependency_returns_guidance(monkeypatch, tmp_path) -> None:
    artifact = _fit_regression(tmp_path)

    def _raise_missing() -> tuple[None, None]:
        raise VeldraValidationError(
            "ONNX export requires optional dependencies. Missing package: 'onnxmltools'. "
            "Install with: uv sync --extra export-onnx"
        )

    monkeypatch.setattr(exporter, "_load_onnx_toolchain", _raise_missing)
    with pytest.raises(VeldraValidationError, match="uv sync --extra export-onnx"):
        export(artifact, format="onnx")


@pytest.mark.parametrize("task_type", ["regression", "frontier"])
def test_export_onnx_generates_model_when_toolchain_available(tmp_path, task_type: str) -> None:
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnxconverter_common")
    artifact = _fit_regression(tmp_path) if task_type == "regression" else _fit_frontier(tmp_path)
    result = export(artifact, format="onnx")
    model_path = Path(result.path) / "model.onnx"
    assert model_path.exists()
