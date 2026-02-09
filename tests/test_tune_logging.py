from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from veldra.api import tune


def _regression_frame(rows: int = 30, seed: int = 512) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.3 * x1 - 0.8 * x2 + rng.normal(scale=0.3, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_tune_emits_progress_logs_with_selected_level(tmp_path, caplog) -> None:
    data_path = tmp_path / "train.csv"
    _regression_frame().to_csv(data_path, index=False)

    with caplog.at_level(logging.DEBUG, logger="veldra"):
        tune(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(data_path), "target": "target"},
                "split": {"type": "kfold", "n_splits": 2, "seed": 9},
                "tuning": {"enabled": True, "n_trials": 1, "log_level": "DEBUG"},
                "export": {"artifact_dir": str(tmp_path / "artifacts")},
            }
        )

    event_messages = [getattr(record, "event_message", "") for record in caplog.records]
    assert "tune trial completed" in event_messages
    assert "tune completed" in event_messages
    trial_records = [
        record
        for record in caplog.records
        if getattr(record, "event_message", "") == "tune trial completed"
    ]
    assert trial_records
    assert all(record.levelno == logging.DEBUG for record in trial_records)
