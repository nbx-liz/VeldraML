from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.notebook_e2e

SUMMARY_FILES = {
    "UC-1": Path("examples/out/phase35_uc01_regression_fit_evaluate/summary.json"),
    "UC-2": Path("examples/out/phase35_uc02_binary_fit_evaluate/summary.json"),
    "UC-3": Path("examples/out/phase35_uc03_multiclass_fit_evaluate/summary.json"),
    "UC-4": Path("examples/out/phase35_uc04_timeseries_fit_evaluate/summary.json"),
    "UC-5": Path("examples/out/phase35_uc05_frontier_fit_evaluate/summary.json"),
    "UC-6": Path("examples/out/phase35_uc06_dr_estimate/summary.json"),
    "UC-7": Path("examples/out/phase35_uc07_drdid_estimate/summary.json"),
    "UC-8": Path("examples/out/phase35_uc08_artifact_evaluate/summary.json"),
    "UC-9": Path("examples/out/phase35_uc09_binary_tune_evaluate/summary.json"),
    "UC-10": Path("examples/out/phase35_uc10_timeseries_tune_evaluate/summary.json"),
    "UC-11": Path("examples/out/phase35_uc11_frontier_tune_evaluate/summary.json"),
    "UC-12": Path("examples/out/phase35_uc12_dr_tune_estimate/summary.json"),
    "UC-13": Path("examples/out/phase35_uc13_drdid_tune_estimate/summary.json"),
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

REQUIRED_CSV_COLUMNS: dict[str, dict[str, set[str]]] = {
    "UC-1": {
        "metrics.csv": {"label", "rmse", "mae", "mape", "r2", "huber"},
        "regression_scores.csv": {"y_true", "prediction", "residual", "in_out_label"},
    },
    "UC-2": {
        "metrics.csv": {
            "label",
            "auc",
            "logloss",
            "brier",
            "average_precision",
            "accuracy",
            "f1",
            "top5_pct_positive",
        },
        "binary_scores.csv": {"y_true", "score", "in_out_label"},
    },
    "UC-3": {
        "metrics.csv": {
            "label",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "multi_logloss",
            "multi_error",
            "brier_macro",
            "ovr_roc_auc_macro",
            "average_precision_macro",
        },
        "multiclass_scores.csv": {"y_true", "fold_id", "in_out_label"},
    },
    "UC-4": {
        "metrics.csv": {"label", "rmse", "mae", "mape", "r2", "huber"},
        "timeseries_detail.csv": {"y_true", "prediction", "residual", "in_out_label"},
    },
    "UC-5": {
        "metrics.csv": {"label", "pinball", "mae", "coverage", "reg_rmse", "reg_huber"},
        "frontier_scores.csv": {"y_true", "prediction", "efficiency"},
    },
    "UC-6": {
        "metrics.csv": {"label", "estimate", "std_error", "overlap_metric"},
        "dr_table.csv": {"treatment", "outcome", "e_hat", "weight"},
        "balance_smd.csv": {"feature", "smd_unweighted", "smd_weighted"},
    },
    "UC-7": {
        "metrics.csv": {"label", "estimate", "std_error", "overlap_metric"},
        "drdid_table.csv": {"treatment", "outcome", "e_hat", "weight"},
        "balance_smd.csv": {"feature", "smd_unweighted", "smd_weighted"},
    },
    "UC-8": {
        "metrics.csv": {"label"},
        "eval_table.csv": {"y_true", "prediction", "residual", "in_out_label"},
        "compare.csv": {"metric", "evaluate_value", "recomputed_value", "delta"},
    },
    "UC-9": {
        "metrics.csv": {
            "label",
            "auc",
            "logloss",
            "brier",
            "average_precision",
            "accuracy",
            "f1",
            "top5_pct_positive",
        },
        "binary_scores.csv": {"y_true", "score", "in_out_label"},
        "tuning_trials.csv": {"number", "value", "state"},
    },
    "UC-10": {
        "metrics.csv": {"label", "rmse", "mae", "mape", "r2", "huber"},
        "timeseries_detail.csv": {"y_true", "prediction", "residual", "in_out_label"},
        "tuning_trials.csv": {"number", "value", "state"},
    },
    "UC-11": {
        "metrics.csv": {"label", "pinball", "mae", "coverage", "reg_rmse", "reg_huber"},
        "frontier_scores.csv": {"y_true", "prediction", "efficiency"},
        "tuning_trials.csv": {"number", "value", "state"},
    },
    "UC-12": {
        "metrics.csv": {"label", "estimate", "std_error", "ci_lower", "ci_upper", "overlap_metric"},
        "dr_table.csv": {"treatment", "outcome", "e_hat", "weight"},
        "balance_smd.csv": {"feature", "smd_unweighted", "smd_weighted"},
        "tuning_trials.csv": {"number", "value", "state"},
    },
    "UC-13": {
        "metrics.csv": {"label", "estimate", "std_error", "ci_lower", "ci_upper", "overlap_metric"},
        "drdid_table.csv": {"treatment", "outcome", "e_hat", "weight"},
        "balance_smd.csv": {"feature", "smd_unweighted", "smd_weighted"},
        "tuning_trials.csv": {"number", "value", "state"},
    },
}


def _assert_metric_bounds(frame: pd.DataFrame) -> None:
    bounded_01 = [
        "auc",
        "accuracy",
        "f1",
        "average_precision",
        "top5_pct_positive",
        "balanced_accuracy",
        "ovr_roc_auc_macro",
        "average_precision_macro",
        "coverage",
    ]
    non_negative = [
        "rmse",
        "mae",
        "mape",
        "huber",
        "pinball",
        "logloss",
        "brier",
        "multi_logloss",
        "multi_error",
        "brier_macro",
        "std_error",
    ]
    for col in bounded_01:
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            assert ((series >= 0.0) & (series <= 1.0)).all(), col
    for col in non_negative:
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            assert (series >= 0.0).all(), col


def test_quick_reference_outputs_have_materialized_files() -> None:
    for uc, summary_path in SUMMARY_FILES.items():
        assert summary_path.exists(), uc
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload.get("uc") == uc
        assert payload.get("status") == "passed", uc

        csv_outputs: dict[str, pd.DataFrame] = {}
        for out in payload.get("outputs", []):
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
