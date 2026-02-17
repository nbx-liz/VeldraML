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

LEGACY_NOTEBOOKS = [
    "phase26_2_uc01_regression_fit_evaluate.ipynb",
    "phase26_2_uc02_binary_tune_evaluate.ipynb",
    "phase26_2_uc03_frontier_fit_evaluate.ipynb",
    "phase26_2_uc04_causal_dr_estimate.ipynb",
    "phase26_2_uc05_causal_drdid_estimate.ipynb",
    "phase26_2_uc06_causal_dr_tune.ipynb",
    "phase26_2_uc07_artifact_evaluate.ipynb",
    "phase26_2_uc08_artifact_reevaluate.ipynb",
    "phase26_2_uc09_export_python_onnx.ipynb",
    "phase26_2_uc10_export_html_excel.ipynb",
    "phase26_3_uc_multiclass_fit_evaluate.ipynb",
    "phase26_3_uc_timeseries_fit_evaluate.ipynb",
    "phase26_2_ux_audit.ipynb",
]


def _load_notebook(path: Path) -> dict:
    assert path.exists(), f"Notebook file is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _cells_source(nb: dict) -> str:
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            chunks.append("".join(src))
        else:
            chunks.append(str(src))
    return "\n".join(chunks)


def test_all_phase26_2_uc_notebooks_exist() -> None:
    for notebook in UC_NOTEBOOKS:
        path = Path("notebooks") / notebook
        assert path.exists(), notebook


def test_phase26_2_uc_notebooks_have_required_sections() -> None:
    for notebook in UC_NOTEBOOKS:
        nb = _load_notebook(Path("notebooks") / notebook)
        source = _cells_source(nb)
        assert "## Overview" in source, notebook
        assert "## Learn More" in source, notebook
        assert "## Setup" in source, notebook
        assert "## Config Notes" in source, notebook
        assert "## Workflow" in source, notebook
        assert "## Result Summary" in source, notebook
        assert "SUMMARY =" in source, notebook


def test_phase26_2_audit_hub_links_all_uc_notebooks() -> None:
    path = Path("notebooks/reference_index.ipynb")
    nb = _load_notebook(path)
    source = _cells_source(nb)
    assert "Notebook Reference Index" in source
    assert "## Tutorials" in source
    assert "## Quick Reference" in source
    for notebook in UC_NOTEBOOKS:
        assert notebook in source, notebook


def test_legacy_phase26_notebooks_are_compatibility_stubs() -> None:
    for notebook in LEGACY_NOTEBOOKS:
        source = _cells_source(_load_notebook(Path("notebooks") / notebook))
        assert "Compatibility Stub" in source, notebook
        assert "Moved to `notebooks/" in source, notebook
