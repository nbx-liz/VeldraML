import json

import numpy as np
import pandas as pd
import pytest

from examples import evaluate_demo_multiclass_artifact
from veldra.api import fit


def _multiclass_frame(rows_per_class: int = 32, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["setosa", "versicolor", "virginica"]
    frames: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        center = float(idx) * 1.7
        x1 = rng.normal(loc=center, scale=0.45, size=rows_per_class)
        x2 = rng.normal(loc=1.0 - center, scale=0.45, size=rows_per_class)
        frames.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(frames, ignore_index=True)


def _train_artifact(tmp_path) -> str:
    train_path = tmp_path / "multiclass_train.csv"
    _multiclass_frame().to_csv(train_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(train_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 9},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert run.artifact_path is not None
    return run.artifact_path


def test_evaluate_demo_multiclass_artifact_writes_metrics(tmp_path) -> None:
    artifact_path = _train_artifact(tmp_path)
    eval_data_path = tmp_path / "multiclass_eval.csv"
    _multiclass_frame(rows_per_class=10, seed=78).to_csv(eval_data_path, index=False)
    out_dir = tmp_path / "eval_out"

    exit_code = evaluate_demo_multiclass_artifact.main(
        [
            "--artifact-path",
            artifact_path,
            "--data-path",
            str(eval_data_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    run_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    eval_only_path = run_dirs[0] / "eval_only_result.json"
    assert eval_only_path.exists()
    payload = json.loads(eval_only_path.read_text(encoding="utf-8"))
    assert {"accuracy", "macro_f1", "logloss"} <= set(payload["metrics"])


def test_evaluate_demo_multiclass_artifact_requires_target_column(tmp_path) -> None:
    artifact_path = _train_artifact(tmp_path)
    bad_data_path = tmp_path / "bad_multiclass_eval.csv"
    _multiclass_frame(rows_per_class=8, seed=79).drop(columns=["target"]).to_csv(
        bad_data_path, index=False
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluate_demo_multiclass_artifact.main(
            [
                "--artifact-path",
                artifact_path,
                "--data-path",
                str(bad_data_path),
            ]
        )

    assert "target" in str(exc_info.value).lower()
