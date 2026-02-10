from __future__ import annotations

from pathlib import Path

import pandas as pd

from veldra.api import estimate_dr


def test_estimate_dr_writes_expected_output_files(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 1.8],
            "x2": [1.0, 0.9, 1.1, 0.8, 0.7, 0.6, 0.4, 0.3],
            "treatment": [0, 0, 1, 0, 1, 1, 1, 0],
            "outcome": [1.0, 1.2, 2.2, 1.4, 2.5, 2.8, 3.0, 1.7],
        }
    )
    train_path = tmp_path / "dr_train.csv"
    frame.to_csv(train_path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(train_path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "causal": {"treatment_col": "treatment"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )

    summary_path = Path(result.metadata["summary_path"])
    obs_path = Path(result.metadata["observation_path"])
    manifest_path = Path(result.metadata["manifest_path"])
    run_config_path = summary_path.parent / "run_config.yaml"

    assert summary_path.exists()
    assert obs_path.exists()
    assert manifest_path.exists()
    assert run_config_path.exists()
