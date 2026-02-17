from __future__ import annotations

import json
from pathlib import Path

UC_NOTEBOOKS = [
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
        assert "## Setup" in source, notebook
        assert "## Workflow" in source, notebook
        assert "## Result Summary" in source, notebook
        assert "SUMMARY =" in source, notebook


def test_phase26_2_audit_hub_links_all_uc_notebooks() -> None:
    path = Path("notebooks/phase26_2_ux_audit.ipynb")
    nb = _load_notebook(path)
    source = _cells_source(nb)
    assert "Phase26.2 UX Audit Hub" in source
    for notebook in UC_NOTEBOOKS:
        assert notebook in source, notebook
