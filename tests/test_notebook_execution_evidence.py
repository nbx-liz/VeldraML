from __future__ import annotations

import json
from pathlib import Path

QUICK_REFERENCE_NOTEBOOKS = [
    "quick_reference/reference_01_regression_fit_evaluate.ipynb",
    "quick_reference/reference_02_binary_tune_evaluate.ipynb",
    "quick_reference/reference_03_frontier_fit_evaluate.ipynb",
    "quick_reference/reference_04_causal_dr_estimate.ipynb",
    "quick_reference/reference_05_causal_drdid_estimate.ipynb",
    "quick_reference/reference_06_causal_dr_tune.ipynb",
    "quick_reference/reference_07_artifact_evaluate.ipynb",
    "quick_reference/reference_08_artifact_reevaluate.ipynb",
    "quick_reference/reference_09_export_python_onnx.ipynb",
    "quick_reference/reference_10_export_html_excel.ipynb",
    "quick_reference/reference_11_multiclass_fit_evaluate.ipynb",
    "quick_reference/reference_12_timeseries_fit_evaluate.ipynb",
]

SUMMARY_FILES = {
    "UC-1": Path("examples/out/phase26_2_uc01_regression_fit_evaluate/summary.json"),
    "UC-2": Path("examples/out/phase26_2_uc02_binary_tune_evaluate/summary.json"),
    "UC-3": Path("examples/out/phase26_2_uc03_frontier_fit_evaluate/summary.json"),
    "UC-4": Path("examples/out/phase26_2_uc04_causal_dr_estimate/summary.json"),
    "UC-5": Path("examples/out/phase26_2_uc05_causal_drdid_estimate/summary.json"),
    "UC-6": Path("examples/out/phase26_2_uc06_causal_dr_tune/summary.json"),
    "UC-7": Path("examples/out/phase26_2_uc07_artifact_evaluate/summary.json"),
    "UC-8": Path("examples/out/phase26_2_uc08_artifact_reevaluate/summary.json"),
    "UC-11": Path("examples/out/phase26_3_uc_multiclass_fit_evaluate/summary.json"),
    "UC-12": Path("examples/out/phase26_3_uc_timeseries_fit_evaluate/summary.json"),
}


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_quick_reference_notebooks_keep_executed_outputs() -> None:
    for notebook in QUICK_REFERENCE_NOTEBOOKS:
        nb = _load_notebook(Path("notebooks") / notebook)
        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        assert code_cells, notebook
        assert all(cell.get("execution_count") is not None for cell in code_cells), notebook
        assert any(cell.get("outputs") for cell in code_cells), notebook


def test_summary_files_have_passed_status_and_materialized_outputs() -> None:
    for uc, summary_path in SUMMARY_FILES.items():
        assert summary_path.exists(), uc
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload.get("uc") == uc
        assert payload.get("status") == "passed", uc
        outputs = payload.get("outputs")
        assert isinstance(outputs, list) and outputs, uc
        for out in outputs:
            path = Path(str(out))
            assert path.exists(), f"{uc}: missing output path {path}"


def test_export_uc_paths_are_materialized() -> None:
    required = [
        Path("examples/out/phase26_2_uc09_export_python_onnx/exports/python"),
        Path("examples/out/phase26_2_uc09_export_python_onnx/exports/onnx"),
        Path("examples/out/phase26_2_uc10_export_html_excel/reports/report.html"),
        Path("examples/out/phase26_2_uc10_export_html_excel/reports/report.xlsx"),
    ]
    for path in required:
        assert path.exists(), path


def test_legacy_manifests_are_removed() -> None:
    assert not Path("notebooks/phase26_2_execution_manifest.json").exists()
    assert not Path("notebooks/phase26_3_execution_manifest.json").exists()
