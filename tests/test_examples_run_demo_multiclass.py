import json

import numpy as np
import pandas as pd
import pytest

from examples import run_demo_multiclass


def _multiclass_frame(rows_per_class: int = 35, seed: int = 66) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["setosa", "versicolor", "virginica"]
    frames: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        center = float(idx) * 2.0
        x1 = rng.normal(loc=center, scale=0.4, size=rows_per_class)
        x2 = rng.normal(loc=1.2 - center, scale=0.4, size=rows_per_class)
        frames.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(frames, ignore_index=True)


def test_run_demo_multiclass_writes_expected_outputs(tmp_path) -> None:
    data_path = tmp_path / "demo_multiclass.csv"
    out_dir = tmp_path / "out"
    _multiclass_frame().to_csv(data_path, index=False)

    exit_code = run_demo_multiclass.main(
        [
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "11",
            "--n-splits",
            "3",
        ]
    )

    assert exit_code == 0
    run_dirs = [path for path in out_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    run_result_path = run_dir / "run_result.json"
    eval_result_path = run_dir / "eval_result.json"
    pred_sample_path = run_dir / "predictions_sample.csv"
    used_config_path = run_dir / "used_config.yaml"

    assert run_result_path.exists()
    assert eval_result_path.exists()
    assert pred_sample_path.exists()
    assert used_config_path.exists()

    run_result = json.loads(run_result_path.read_text(encoding="utf-8"))
    artifact_path = run_result["artifact_path"]
    assert artifact_path
    assert (run_dir / "artifacts").exists()

    pred_sample = pd.read_csv(pred_sample_path)
    assert "label_pred" in pred_sample.columns
    assert len([c for c in pred_sample.columns if c.startswith("proba_")]) >= 3


def test_run_demo_multiclass_requires_prepared_csv(tmp_path) -> None:
    missing_path = tmp_path / "missing_multiclass.csv"
    with pytest.raises(SystemExit) as exc_info:
        run_demo_multiclass.main(["--data-path", str(missing_path)])

    assert "prepare_demo_data_multiclass.py" in str(exc_info.value)
