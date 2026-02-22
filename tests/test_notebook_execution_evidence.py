from __future__ import annotations

import json
from pathlib import Path

QUICK_REFERENCE_NOTEBOOKS = [
    "quick_reference/reference_01_regression_fit_evaluate.ipynb",
    "quick_reference/reference_02_binary_fit_evaluate.ipynb",
    "quick_reference/reference_03_multiclass_fit_evaluate.ipynb",
    "quick_reference/reference_04_timeseries_fit_evaluate.ipynb",
    "quick_reference/reference_05_frontier_fit_evaluate.ipynb",
    "quick_reference/reference_06_dr_estimate.ipynb",
    "quick_reference/reference_07_drdid_estimate.ipynb",
    "quick_reference/reference_08_artifact_evaluate.ipynb",
    "quick_reference/reference_09_binary_tune_evaluate.ipynb",
    "quick_reference/reference_10_timeseries_tune_evaluate.ipynb",
    "quick_reference/reference_11_frontier_tune_evaluate.ipynb",
    "quick_reference/reference_12_dr_tune_estimate.ipynb",
    "quick_reference/reference_13_drdid_tune_estimate.ipynb",
]

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
