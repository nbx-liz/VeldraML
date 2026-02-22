from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = [
    "quick_reference/reference_01_regression_fit_evaluate.ipynb",
    "quick_reference/reference_02_binary_tune_evaluate.ipynb",
    "quick_reference/reference_03_frontier_fit_evaluate.ipynb",
    "quick_reference/reference_04_causal_dr_estimate.ipynb",
    "quick_reference/reference_05_causal_drdid_estimate.ipynb",
    "quick_reference/reference_06_causal_dr_tune.ipynb",
    "quick_reference/reference_07_artifact_evaluate.ipynb",
    "quick_reference/reference_08_artifact_reevaluate.ipynb",
    "quick_reference/reference_11_multiclass_fit_evaluate.ipynb",
    "quick_reference/reference_12_timeseries_fit_evaluate.ipynb",
]
UC1_NOTEBOOK = "quick_reference/reference_01_regression_fit_evaluate.ipynb"

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

REMOVED_LEGACY_NOTEBOOKS = [
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


def _load(path: Path) -> dict:
    assert path.exists(), f"Notebook file is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path) -> str:
    payload = _load(path)
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        src = cell.get("source", [])
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def test_quickref_notebook_contract() -> None:
    for nb in NOTEBOOKS:
        path = Path("notebooks") / nb
        payload = _load(path)
        src = _source(path)
        if nb != UC1_NOTEBOOK:
            assert "## Overview" in src
            assert "## Learn More" in src
        assert "## Setup" in src
        if nb != UC1_NOTEBOOK:
            assert "## Config Notes" in src
            assert "## Workflow" in src
        if nb != UC1_NOTEBOOK:
            assert "### Output Annotation" in src
        assert "## Result Summary" in src
        assert "SUMMARY =" in src
        if nb == UC1_NOTEBOOK:
            assert "matplotlib.use('Agg')" not in src
            assert "plot_learning_curve(" in src
        else:
            assert "matplotlib.use('Agg')" in src
        assert "from veldra.diagnostics import" in src
        assert "metrics_df" in src
        assert "display(" in src
        assert "placeholder" not in src.lower()

        code_cells = [cell for cell in payload.get("cells", []) if cell.get("cell_type") == "code"]
        assert code_cells, nb
        assert all(cell.get("execution_count") is not None for cell in code_cells), nb
        assert any(cell.get("outputs") for cell in code_cells), nb

        if nb == UC1_NOTEBOOK:
            assert len(payload.get("cells", [])) == 31, nb


def test_quick_reference_notebooks_exist() -> None:
    for notebook in UC_NOTEBOOKS:
        assert (Path("notebooks") / notebook).exists(), notebook


def test_quick_reference_notebooks_have_required_sections() -> None:
    for notebook in UC_NOTEBOOKS:
        source = _source(Path("notebooks") / notebook)
        if notebook != UC1_NOTEBOOK:
            assert "## Overview" in source, notebook
            assert "## Learn More" in source, notebook
        assert "## Setup" in source, notebook
        if notebook != UC1_NOTEBOOK:
            assert "## Config Notes" in source, notebook
            assert "## Workflow" in source, notebook
        assert "## Result Summary" in source, notebook
        assert "SUMMARY" in source, notebook


def test_reference_index_links_all_quick_reference_notebooks() -> None:
    source = _source(Path("notebooks/reference_index.ipynb"))
    assert "Notebook Reference Index" in source
    assert "## Tutorials" in source
    assert "## Quick Reference" in source
    for notebook in UC_NOTEBOOKS:
        assert notebook in source, notebook


def test_legacy_phase_notebooks_are_removed() -> None:
    for notebook in REMOVED_LEGACY_NOTEBOOKS:
        assert not (Path("notebooks") / notebook).exists(), notebook
