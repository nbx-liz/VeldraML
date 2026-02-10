from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from veldra.api import Artifact, export, fit


def _regression_frame(rows: int = 40, seed: int = 1201) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.4 * x1 - 0.8 * x2 + rng.normal(scale=0.2, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _binary_frame(rows: int = 50, seed: int = 1202) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = x1 - x2 + rng.normal(scale=0.3, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _multiclass_frame(rows_per_class: int = 12, seed: int = 1203) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["alpha", "beta", "gamma"]
    chunks: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        x1 = rng.normal(loc=idx * 1.3, scale=0.35, size=rows_per_class)
        x2 = rng.normal(loc=1.5 - idx * 1.0, scale=0.35, size=rows_per_class)
        chunks.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(chunks, ignore_index=True)


def _frontier_frame(rows: int = 60, seed: int = 1204) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    y = 1.0 + 1.5 * x1 - 0.4 * x2 + rng.normal(scale=0.2, size=rows) + rng.exponential(
        scale=0.25, size=rows
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _fit_payload(task: str, data_path: Path, artifact_dir: Path) -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task},
        "data": {"path": str(data_path), "target": "target"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 42},
        "export": {"artifact_dir": str(artifact_dir)},
    }
    if task == "binary":
        payload["split"] = {"type": "stratified", "n_splits": 2, "seed": 42}
        payload["postprocess"] = {"calibration": "platt"}
    if task == "multiclass":
        payload["split"] = {"type": "stratified", "n_splits": 3, "seed": 42}
    if task == "frontier":
        payload["frontier"] = {"alpha": 0.90}
    return payload


@pytest.mark.parametrize(
    ("task", "frame_factory"),
    [
        ("regression", _regression_frame),
        ("binary", _binary_frame),
        ("multiclass", _multiclass_frame),
        ("frontier", _frontier_frame),
    ],
)
def test_export_python_mvp_creates_expected_files(
    task: str,
    frame_factory,
    tmp_path,
) -> None:
    frame = frame_factory()
    data_path = tmp_path / f"{task}.csv"
    frame.to_csv(data_path, index=False)
    run = fit(_fit_payload(task, data_path, tmp_path))
    artifact = Artifact.load(run.artifact_path)

    export_result = export(artifact, format="python")
    export_dir = Path(export_result.path)
    assert export_result.format == "python"
    assert export_dir.exists()
    assert {"manifest.json", "run_config.yaml", "feature_schema.json", "model.lgb.txt"} <= {
        p.name for p in export_dir.iterdir()
    }
    assert {"metadata.json", "runtime_predict.py", "README.md"} <= {
        p.name for p in export_dir.iterdir()
    }
