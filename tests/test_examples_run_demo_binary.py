import json

import numpy as np
import pandas as pd
import pytest

from examples import run_demo_binary


def _binary_frame(rows: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(loc=1.0, scale=0.8, size=rows)
    logit = 1.5 * x1 - 1.1 * x2 + rng.normal(scale=0.4, size=rows)
    y = (logit > np.median(logit)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_run_demo_binary_writes_expected_outputs(tmp_path) -> None:
    data_path = tmp_path / "demo_binary.csv"
    out_dir = tmp_path / "out"
    _binary_frame().to_csv(data_path, index=False)

    exit_code = run_demo_binary.main(
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
    assert {"target", "p_cal", "p_raw", "label_pred"} <= set(pred_sample.columns)


def test_run_demo_binary_requires_prepared_csv(tmp_path) -> None:
    missing_path = tmp_path / "missing_binary.csv"
    with pytest.raises(SystemExit) as exc_info:
        run_demo_binary.main(["--data-path", str(missing_path)])

    assert "prepare_demo_data_binary.py" in str(exc_info.value)
