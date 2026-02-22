from __future__ import annotations

import json
import re
from pathlib import Path

QR = Path("notebooks/quick_reference")

A_NOTEBOOK_METRICS = {
    QR / "reference_01_regression_fit_evaluate.ipynb": ["rmse", "mae"],
    QR / "reference_02_binary_fit_evaluate.ipynb": ["logloss", "auc"],
    QR / "reference_03_multiclass_fit_evaluate.ipynb": ["multi_logloss", "multi_error"],
    QR / "reference_04_timeseries_fit_evaluate.ipynb": ["rmse", "mae"],
    QR / "reference_05_frontier_fit_evaluate.ipynb": ["quantile"],
    QR / "reference_06_dr_estimate.ipynb": ["rmse", "mae"],
    QR / "reference_07_drdid_estimate.ipynb": ["rmse", "mae"],
}

A_COMMON_PATTERNS = [
    r'"num_boost_round"\s*:\s*1200',
    r'"early_stopping_rounds"\s*:\s*120',
    r'"early_stopping_validation_fraction"\s*:\s*0\.2',
    r'"auto_num_leaves"\s*:\s*true',
    r'"num_leaves_ratio"\s*:\s*1(?:\.0)?',
    r'"min_data_in_leaf_ratio"\s*:\s*0\.01',
    r'"min_data_in_bin_ratio"\s*:\s*0\.01',
    r'"first_metric_only"\s*:\s*true',
]

B_NOTEBOOK_OBJECTIVES = {
    QR / "reference_09_binary_tune_evaluate.ipynb": "brier",
    QR / "reference_10_timeseries_tune_evaluate.ipynb": "rmse",
    QR / "reference_11_frontier_tune_evaluate.ipynb": "pinball_coverage_penalty",
    QR / "reference_12_dr_tune_estimate.ipynb": "dr_balance_priority",
    QR / "reference_13_drdid_tune_estimate.ipynb": "drdid_balance_priority",
}

B_COMMON_PATTERNS = [
    r'"n_trials"\s*:\s*3',
    r'"resume"\s*:\s*true',
    r'"preset"\s*:\s*"standard"',
]


def _notebook_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_chunks: list[str] = []
    for cell in payload.get("cells", []):
        raw = cell.get("source", [])
        source_chunks.append("".join(raw) if isinstance(raw, list) else str(raw))
    return "\n".join(source_chunks).lower().replace("'", '"')


def _assert_patterns(source: str, patterns: list[str], notebook: Path) -> None:
    for pattern in patterns:
        assert re.search(pattern, source), f"{notebook}: missing pattern {pattern}"


def _metrics_pattern(metrics: list[str]) -> str:
    ordered = ",\\s*".join(fr'"{re.escape(metric)}"' for metric in metrics)
    return fr'"metrics"\s*:\s*\[{ordered}\]'


def test_reference_a_contract_applied_to_phase35_quickref() -> None:
    for notebook, metrics in A_NOTEBOOK_METRICS.items():
        source = _notebook_source(notebook)
        _assert_patterns(source, A_COMMON_PATTERNS, notebook)
        _assert_patterns(source, [_metrics_pattern(metrics)], notebook)


def test_reference_b_contract_applied_to_tuning_notebooks() -> None:
    for notebook, objective in B_NOTEBOOK_OBJECTIVES.items():
        source = _notebook_source(notebook)
        objective_pattern = fr'"objective"\s*:\s*"{re.escape(objective)}"'
        _assert_patterns(source, [objective_pattern], notebook)
        _assert_patterns(source, B_COMMON_PATTERNS, notebook)
