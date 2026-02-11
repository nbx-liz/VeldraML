from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from veldra.api import estimate_dr


def test_drdid_writes_summary_and_observation_outputs(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [0, 1, 0, 1, 0, 1, 0, 1],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "treatment": [0, 0, 1, 1, 0, 0, 1, 1],
            "x1": [0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4],
            "outcome": [10.0, 11.0, 12.0, 14.5, 9.5, 10.3, 13.0, 15.1],
        }
    )
    path = tmp_path / "panel.csv"
    frame.to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 5},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
            },
            "export": {"artifact_dir": str(tmp_path)},
        }
    )

    summary_path = Path(result.metadata["summary_path"])
    obs_path = Path(result.metadata["observation_path"])
    assert summary_path.exists()
    assert obs_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["method"] == "dr_did"
    assert payload["design"] == "panel"
    assert payload["n_pre"] == 4
    assert payload["n_post"] == 4

