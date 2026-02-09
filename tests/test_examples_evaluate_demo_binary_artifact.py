import json

import numpy as np
import pandas as pd
import pytest

from examples import evaluate_demo_binary_artifact
from veldra.api import fit


def _binary_frame(rows: int = 100, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    logit = 1.8 * x1 - 1.0 * x2 + rng.normal(scale=0.5, size=rows)
    y = (logit > np.median(logit)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _train_binary_artifact(tmp_path) -> str:
    train_path = tmp_path / "binary_train.csv"
    _binary_frame().to_csv(train_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(train_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 9},
            "postprocess": {"calibration": "platt", "threshold": 0.5},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert run.artifact_path is not None
    return run.artifact_path


def test_evaluate_demo_binary_artifact_writes_metrics(tmp_path) -> None:
    artifact_path = _train_binary_artifact(tmp_path)
    eval_data_path = tmp_path / "binary_eval.csv"
    _binary_frame(rows=40, seed=22).to_csv(eval_data_path, index=False)
    out_dir = tmp_path / "eval_out"

    exit_code = evaluate_demo_binary_artifact.main(
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
    assert set(payload["metrics"]) == {"auc", "logloss", "brier"}


def test_evaluate_demo_binary_artifact_requires_target_column(tmp_path) -> None:
    artifact_path = _train_binary_artifact(tmp_path)
    bad_data_path = tmp_path / "bad_binary_eval.csv"
    _binary_frame(rows=20, seed=23).drop(columns=["target"]).to_csv(bad_data_path, index=False)

    with pytest.raises(SystemExit) as exc_info:
        evaluate_demo_binary_artifact.main(
            [
                "--artifact-path",
                artifact_path,
                "--data-path",
                str(bad_data_path),
            ]
        )

    assert "target" in str(exc_info.value).lower()
