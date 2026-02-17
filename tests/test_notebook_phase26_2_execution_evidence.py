from __future__ import annotations

import json
from pathlib import Path

UC_NOTEBOOKS = [
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
]


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_uc_notebooks_keep_executed_outputs() -> None:
    for notebook in UC_NOTEBOOKS:
        nb = _load_notebook(Path("notebooks") / notebook)
        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        assert code_cells, notebook
        assert all(cell.get("execution_count") is not None for cell in code_cells), notebook
        assert any(cell.get("outputs") for cell in code_cells), notebook


def test_execution_manifest_has_10_entries_with_passed_status() -> None:
    path = Path("notebooks/phase26_2_execution_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    assert isinstance(entries, list)
    assert len(entries) == 10

    uc_values = {entry.get("uc") for entry in entries}
    assert uc_values == {f"UC-{i}" for i in range(1, 11)}

    for entry in entries:
        assert entry.get("status") == "passed", entry.get("uc")
        outputs = entry.get("outputs")
        assert isinstance(outputs, list)
        assert outputs, entry.get("uc")


def test_manifest_output_paths_are_materialized_when_present() -> None:
    payload = json.loads(
        Path("notebooks/phase26_2_execution_manifest.json").read_text(encoding="utf-8")
    )
    for entry in payload.get("entries", []):
        for out in entry.get("outputs", []):
            if not out:
                continue
            path = Path(str(out))
            assert path.exists(), f"{entry.get('uc')}: missing output path {path}"
