from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from examples import run_demo_export
from veldra.api import fit


def _fit_artifact(tmp_path) -> str:
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
    return run.artifact_path


def test_run_demo_export_writes_result_json(tmp_path) -> None:
    artifact_path = _fit_artifact(tmp_path)
    out_dir = tmp_path / "out"

    code = run_demo_export.main(
        [
            "--artifact-path",
            artifact_path,
            "--format",
            "python",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    result_json = run_dirs[0] / "export_result.json"
    assert result_json.exists()

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    export_path = Path(payload["path"])
    assert payload["format"] == "python"
    assert export_path.exists()
