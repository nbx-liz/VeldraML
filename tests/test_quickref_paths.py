from __future__ import annotations

import json
from pathlib import Path

UC_NOTEBOOKS = {
    "quick_reference/reference_01_regression_fit_evaluate.ipynb": ["fit(", "evaluate("],
    "quick_reference/reference_02_binary_tune_evaluate.ipynb": ["tune(", "evaluate("],
    "quick_reference/reference_03_frontier_fit_evaluate.ipynb": ["fit(", "frontier"],
    "quick_reference/reference_04_causal_dr_estimate.ipynb": ["estimate_dr(", "dr"],
    "quick_reference/reference_05_causal_drdid_estimate.ipynb": ["estimate_dr(", "dr_did"],
    "quick_reference/reference_06_causal_dr_tune.ipynb": ["tune(", "dr_balance_priority"],
    "quick_reference/reference_07_artifact_evaluate.ipynb": ["Artifact.load", "evaluate("],
    "quick_reference/reference_08_artifact_reevaluate.ipynb": [
        "_cb_result_eval_precheck",
        "evaluate(",
    ],
    "quick_reference/reference_09_export_python_onnx.ipynb": ["export(", "onnx"],
    "quick_reference/reference_10_export_html_excel.ipynb": [
        "export_html_report",
        "export_excel_report",
    ],
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


def test_uc_notebooks_reference_quickref_out_dirs() -> None:
    for notebook in UC_NOTEBOOKS:
        source = _notebook_source(Path("notebooks") / notebook)
        assert "OUT_DIR = ROOT / 'examples' / 'out' / 'phase26_2_" in source, notebook


def test_uc_notebooks_include_expected_action_snippets() -> None:
    for notebook, snippets in UC_NOTEBOOKS.items():
        source = _notebook_source(Path("notebooks") / notebook)
        for snippet in snippets:
            assert snippet in source, f"{notebook}: {snippet}"
