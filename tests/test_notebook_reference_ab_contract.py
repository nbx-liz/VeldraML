from __future__ import annotations

# ruff: noqa: E501
import json
import re
from pathlib import Path

QR = "notebooks/quick_reference"
TT = "notebooks/tutorials"

A_NOTEBOOK_METRICS = {
    f"{QR}/reference_01_regression_fit_evaluate.ipynb": ["rmse", "mae"],
    f"{QR}/reference_02_binary_tune_evaluate.ipynb": ["logloss", "auc"],
    f"{QR}/reference_03_frontier_fit_evaluate.ipynb": ["quantile"],
    f"{QR}/reference_04_causal_dr_estimate.ipynb": ["rmse", "mae"],
    f"{QR}/reference_05_causal_drdid_estimate.ipynb": ["rmse", "mae"],
    f"{QR}/reference_06_causal_dr_tune.ipynb": ["rmse", "mae"],
    f"{QR}/reference_11_multiclass_fit_evaluate.ipynb": ["multi_logloss", "multi_error"],
    f"{QR}/reference_12_timeseries_fit_evaluate.ipynb": ["rmse", "mae"],
    f"{TT}/tutorial_01_regression_basics.ipynb": ["rmse", "mae"],
    f"{TT}/tutorial_02_binary_classification_tuning.ipynb": ["logloss", "auc"],
    f"{TT}/tutorial_03_frontier_quantile_regression.ipynb": ["quantile"],
    f"{TT}/tutorial_04_scenario_simulation.ipynb": ["rmse", "mae"],
    f"{TT}/tutorial_05_causal_dr_lalonde.ipynb": ["rmse", "mae"],
    f"{TT}/tutorial_06_causal_drdid_lalonde.ipynb": ["rmse", "mae"],
}

A_COMMON_PATTERNS = [
    r'"num_boost_round"\s*:\s*2000',
    r'"early_stopping_rounds"\s*:\s*200',
    r'"early_stopping_validation_fraction"\s*:\s*0\.2',
    r'"auto_num_leaves"\s*:\s*true',
    r'"num_leaves_ratio"\s*:\s*1(?:\.0)?',
    r'"min_data_in_leaf_ratio"\s*:\s*0\.01',
    r'"min_data_in_bin_ratio"\s*:\s*0\.01',
    r'"learning_rate"\s*:\s*0\.01',
    r'"max_bin"\s*:\s*255',
    r'"max_depth"\s*:\s*10',
    r'"feature_fraction"\s*:\s*1(?:\.0)?',
    r'"bagging_fraction"\s*:\s*1(?:\.0)?',
    r'"bagging_freq"\s*:\s*0',
    r'"lambda_l1"\s*:\s*0(?:\.0)?',
    r'"lambda_l2"\s*:\s*0\.000001',
    r'"min_child_samples"\s*:\s*20',
    r'"first_metric_only"\s*:\s*true',
]

B_NOTEBOOK_OBJECTIVES = {
    f"{QR}/reference_02_binary_tune_evaluate.ipynb": "brier",
    f"{QR}/reference_06_causal_dr_tune.ipynb": "dr_balance_priority",
    f"{TT}/tutorial_02_binary_classification_tuning.ipynb": "brier",
}

B_COMMON_PATTERNS = [
    r'"learning_rate"\s*:\s*\{"type"\s*:\s*"float",\s*"low"\s*:\s*0\.01,\s*"high"\s*:\s*0\.1,\s*"log"\s*:\s*true\}',
    r'"train\.num_leaves_ratio"\s*:\s*\{"type"\s*:\s*"float",\s*"low"\s*:\s*0\.5,\s*"high"\s*:\s*1\.0\}',
    r'"train\.early_stopping_validation_fraction"\s*:\s*\{"type"\s*:\s*"float",\s*"low"\s*:\s*0\.1,\s*"high"\s*:\s*0\.3\}',
    r'"max_bin"\s*:\s*\{"type"\s*:\s*"int",\s*"low"\s*:\s*127,\s*"high"\s*:\s*255\}',
    r'"train\.min_data_in_leaf_ratio"\s*:\s*\{"type"\s*:\s*"float",\s*"low"\s*:\s*0\.01,\s*"high"\s*:\s*0\.1\}',
    r'"train\.min_data_in_bin_ratio"\s*:\s*\{"type"\s*:\s*"float",\s*"low"\s*:\s*0\.01,\s*"high"\s*:\s*0\.1\}',
    r'"max_depth"\s*:\s*\{"type"\s*:\s*"int",\s*"low"\s*:\s*3,\s*"high"\s*:\s*15\}',
    r'"feature_fraction"\s*:\s*\{"type"\s*:\s*"float",\s*"low"\s*:\s*0\.5,\s*"high"\s*:\s*1\.0\}',
    r'"bagging_fraction"\s*:\s*1(?:\.0)?',
    r'"bagging_freq"\s*:\s*0',
    r'"lambda_l1"\s*:\s*0(?:\.0)?',
    r'"lambda_l2"\s*:\s*\{"type"\s*:\s*"float",\s*"low"\s*:\s*0\.000001,\s*"high"\s*:\s*0\.1\}',
]


def _notebook_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_chunks: list[str] = []
    for cell in payload.get("cells", []):
        raw = cell.get("source", [])
        source_chunks.append("".join(raw) if isinstance(raw, list) else str(raw))
    return "\n".join(source_chunks).lower().replace("'", '"')


def _assert_patterns(source: str, patterns: list[str], notebook: str) -> None:
    for pattern in patterns:
        assert re.search(pattern, source), f"{notebook}: missing pattern {pattern}"


def _metrics_pattern(metrics: list[str]) -> str:
    ordered = ",\\s*".join(fr'"{re.escape(metric)}"' for metric in metrics)
    return fr'"metrics"\s*:\s*\[{ordered}\]'


def test_reference_a_contract_applied_to_quickref_and_tutorials() -> None:
    for notebook, metrics in A_NOTEBOOK_METRICS.items():
        source = _notebook_source(Path(notebook))
        _assert_patterns(source, A_COMMON_PATTERNS, notebook)
        _assert_patterns(source, [_metrics_pattern(metrics)], notebook)


def test_reference_b_contract_applied_to_tuning_notebooks() -> None:
    for notebook, objective in B_NOTEBOOK_OBJECTIVES.items():
        source = _notebook_source(Path(notebook))
        objective_pattern = fr'"objective"\s*:\s*"{re.escape(objective)}"'
        _assert_patterns(source, [objective_pattern], notebook)
        _assert_patterns(source, B_COMMON_PATTERNS, notebook)
