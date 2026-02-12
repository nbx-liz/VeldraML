from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from veldra.api import estimate_dr


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 0.4, 0.7, 0.9, 1.2, 1.4, 1.6, 1.8],
            "x2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2],
            "treatment": [0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            "outcome": [1.0, 1.1, 1.2, 2.0, 1.4, 2.5, 2.6, 2.8, 3.0, 1.9],
        }
    )


def test_estimate_dr_reports_balance_metrics(tmp_path) -> None:
    path = tmp_path / "dr.csv"
    _frame().to_csv(path, index=False)

    result = estimate_dr(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(path), "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "causal": {"method": "dr", "treatment_col": "treatment"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )

    assert {"overlap_metric", "smd_max_unweighted", "smd_max_weighted"} <= set(result.metrics)
    summary = json.loads(Path(result.metadata["summary_path"]).read_text(encoding="utf-8"))
    assert "overlap_metric" in summary
    assert "smd_max_unweighted" in summary
    assert "smd_max_weighted" in summary
