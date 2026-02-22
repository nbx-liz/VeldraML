from __future__ import annotations

import json
from pathlib import Path

QUICKREF_NOTEBOOKS = {
    "quick_reference/reference_01_regression_fit_evaluate.ipynb": {
        "out_dir": "phase35_uc01_regression_fit_evaluate",
        "snippets": ["fit(", "evaluate(", "regression_metrics(", "plot_learning_curve("],
    },
    "quick_reference/reference_02_binary_fit_evaluate.ipynb": {
        "out_dir": "phase35_uc02_binary_fit_evaluate",
        "snippets": [
            "fit(",
            "evaluate(",
            "binary_metrics(",
            "plot_roc_comparison(",
            "plot_confusion_matrix(",
        ],
    },
    "quick_reference/reference_03_multiclass_fit_evaluate.ipynb": {
        "out_dir": "phase35_uc03_multiclass_fit_evaluate",
        "snippets": [
            "fit(",
            "evaluate(",
            "multiclass_metrics(",
            "plot_roc_multiclass(",
            "plot_confusion_matrix(",
        ],
    },
    "quick_reference/reference_04_timeseries_fit_evaluate.ipynb": {
        "out_dir": "phase35_uc04_timeseries_fit_evaluate",
        "snippets": [
            "fit(",
            "evaluate(",
            "regression_metrics(",
            "plot_timeseries_prediction(",
            "plot_timeseries_residual(",
            "'type': 'timeseries'",
        ],
    },
    "quick_reference/reference_05_frontier_fit_evaluate.ipynb": {
        "out_dir": "phase35_uc05_frontier_fit_evaluate",
        "snippets": [
            "fit(",
            "evaluate(",
            "frontier_metrics(",
            "build_frontier_table(",
            "plot_pinball_histogram(",
            "plot_frontier_scatter(",
        ],
    },
    "quick_reference/reference_06_dr_estimate.ipynb": {
        "out_dir": "phase35_uc06_dr_estimate",
        "snippets": [
            "estimate_dr(",
            "build_dr_table(",
            "compute_balance_smd(",
            "plot_love_plot(",
        ],
    },
    "quick_reference/reference_07_drdid_estimate.ipynb": {
        "out_dir": "phase35_uc07_drdid_estimate",
        "snippets": [
            "estimate_dr(",
            "'dr_did'",
            "build_drdid_table(",
            "plot_parallel_trends(",
        ],
    },
    "quick_reference/reference_08_artifact_evaluate.ipynb": {
        "out_dir": "phase35_uc08_artifact_evaluate",
        "snippets": [
            "Artifact.load(",
            "evaluate(",
            "build_regression_table(",
            "regression_metrics(",
            "phase35_uc11_frontier_tune_evaluate",
            "compare.csv",
        ],
    },
    "quick_reference/reference_09_binary_tune_evaluate.ipynb": {
        "out_dir": "phase35_uc09_binary_tune_evaluate",
        "snippets": [
            "tune(",
            "best_params",
            "fit(",
            "binary_metrics(",
            "plot_confusion_matrix(",
        ],
    },
    "quick_reference/reference_10_timeseries_tune_evaluate.ipynb": {
        "out_dir": "phase35_uc10_timeseries_tune_evaluate",
        "snippets": [
            "tune(",
            "best_params",
            "plot_timeseries_prediction(",
            "plot_timeseries_residual(",
            "'type': 'timeseries'",
        ],
    },
    "quick_reference/reference_11_frontier_tune_evaluate.ipynb": {
        "out_dir": "phase35_uc11_frontier_tune_evaluate",
        "snippets": [
            "tune(",
            "best_params",
            "frontier_metrics(",
            "build_frontier_table(",
            "latest_artifact_path.txt",
        ],
    },
    "quick_reference/reference_12_dr_tune_estimate.ipynb": {
        "out_dir": "phase35_uc12_dr_tune_estimate",
        "snippets": [
            "tune(",
            "best_params",
            "estimate_dr(",
            "build_dr_table(",
            "plot_love_plot(",
        ],
    },
    "quick_reference/reference_13_drdid_tune_estimate.ipynb": {
        "out_dir": "phase35_uc13_drdid_tune_estimate",
        "snippets": [
            "tune(",
            "best_params",
            "'dr_did'",
            "build_drdid_table(",
            "plot_parallel_trends(",
        ],
    },
}


def _notebook_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            chunks.append("".join(src))
        else:
            chunks.append(str(src))
    return "\n".join(chunks)


def test_quickref_out_dirs_are_phase35_specific() -> None:
    for notebook, spec in QUICKREF_NOTEBOOKS.items():
        source = _notebook_source(Path("notebooks") / notebook)
        expected = f"OUT_DIR = ROOT / 'examples' / 'out' / '{spec['out_dir']}'"
        assert expected in source, notebook


def test_quickref_include_expected_action_snippets() -> None:
    for notebook, spec in QUICKREF_NOTEBOOKS.items():
        source = _notebook_source(Path("notebooks") / notebook)
        for snippet in spec["snippets"]:
            assert snippet in source, f"{notebook}: {snippet}"
