from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.notebook_e2e

TARGET_UCS = {f"UC-{i}" for i in range(1, 9)} | {"UC-11", "UC-12"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

REQUIRED_CSV_COLUMNS: dict[str, dict[str, set[str]]] = {
    "UC-1": {
        "metrics.csv": {"label", "rmse", "mae", "mape", "r2"},
        "regression_scores.csv": {"y_true", "prediction", "residual", "in_out_label"},
    },
    "UC-2": {
        "metrics.csv": {"label", "auc", "logloss", "brier", "accuracy", "f1"},
        "binary_scores.csv": {"y_true", "score", "in_out_label"},
    },
    "UC-3": {
        "metrics.csv": {"label", "pinball", "mae", "coverage"},
        "frontier_scores.csv": {"y_true", "prediction", "efficiency"},
    },
    "UC-4": {
        "metrics.csv": {"estimate", "std_error", "overlap_metric"},
        "dr_table.csv": {"treatment", "outcome", "e_hat", "weight"},
    },
    "UC-5": {
        "metrics.csv": {"estimate", "std_error", "overlap_metric"},
        "drdid_table.csv": {"treatment", "outcome", "e_hat", "weight"},
    },
    "UC-6": {
        "metrics.csv": {"best_score", "best_param_count"},
        "tuning_trials.csv": {"number", "value", "state"},
    },
    "UC-7": {
        "metrics.csv": {"label", "rmse", "mae", "r2"},
        "eval_table.csv": {"y_true", "prediction", "residual"},
    },
    "UC-8": {
        "reeval_compare.csv": {"metric", "train_value", "reeval_value", "delta"},
        "precheck.csv": {"ok", "missing_columns", "required_columns"},
    },
    "UC-11": {
        "metrics.csv": {"label", "accuracy", "macro_f1", "multi_logloss", "multi_error"},
        "multiclass_scores.csv": {"y_true", "fold_id", "in_out_label"},
    },
    "UC-12": {
        "timeseries_scores.csv": {"label", "rmse", "mae", "mape", "r2"},
        "timeseries_detail.csv": {"y_true", "prediction", "residual"},
    },
}


def _assert_metric_bounds(frame: pd.DataFrame) -> None:
    bounded_01 = ["auc", "accuracy", "f1", "average_precision", "multi_error", "coverage"]
    non_negative = ["rmse", "mae", "mape", "logloss", "brier", "pinball", "std_error"]
    for col in bounded_01:
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            assert ((series >= 0.0) & (series <= 1.0)).all(), col
    for col in non_negative:
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            assert (series >= 0.0).all(), col


def test_phase26_3_outputs_have_materialized_files() -> None:
    payload = json.loads(Path("notebooks/phase26_3_execution_manifest.json").read_text("utf-8"))
    for entry in payload.get("entries", []):
        uc = str(entry.get("uc"))
        if uc not in TARGET_UCS:
            continue

        csv_outputs: dict[str, pd.DataFrame] = {}
        for out in entry.get("outputs", []):
            path = Path(str(out))
            assert path.exists(), path
            assert path.stat().st_size > 0
            if path.suffix.lower() == ".png":
                assert path.read_bytes()[:8] == PNG_SIGNATURE, path
            if path.suffix == ".csv":
                frame = pd.read_csv(path)
                assert not frame.empty
                csv_outputs[path.name] = frame

        for csv_name, required_columns in REQUIRED_CSV_COLUMNS.get(uc, {}).items():
            assert csv_name in csv_outputs, f"{uc}: {csv_name}"
            frame = csv_outputs[csv_name]
            assert required_columns.issubset(set(frame.columns)), f"{uc}: {csv_name}"
            _assert_metric_bounds(frame)
