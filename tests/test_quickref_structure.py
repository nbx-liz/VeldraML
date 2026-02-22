from __future__ import annotations

import json
import re
from pathlib import Path

QUICKREF_NOTEBOOKS = [
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


def test_quick_reference_notebook_contract() -> None:
    for notebook in QUICKREF_NOTEBOOKS:
        path = Path("notebooks") / notebook
        payload = _load(path)
        src = _source(path)

        assert "## Setup" in src
        assert "## Result Summary" in src
        assert "SUMMARY =" in src
        assert "from veldra.diagnostics import" in src
        assert src.count("## ") >= 12, notebook
        assert len(payload.get("cells", [])) >= 25, notebook

        code_cells = [cell for cell in payload.get("cells", []) if cell.get("cell_type") == "code"]
        assert code_cells, notebook
        assert all(cell.get("execution_count") is not None for cell in code_cells), notebook
        assert any(cell.get("outputs") for cell in code_cells), notebook


def test_quick_reference_notebooks_exist() -> None:
    for notebook in QUICKREF_NOTEBOOKS:
        assert (Path("notebooks") / notebook).exists(), notebook


def test_reference_index_links_quick_reference_notebooks() -> None:
    source = _source(Path("notebooks/reference_index.ipynb"))
    assert "Notebook Reference Index" in source
    assert "## Tutorials" in source
    assert "## Quick Reference (Phase35 Main)" in source
    assert "deprecated compatibility alias" in source
    for notebook in QUICKREF_NOTEBOOKS:
        assert notebook in source, notebook


def test_quick_reference_markdown_cells_are_english_only() -> None:
    jp_pattern = re.compile(r"[ぁ-んァ-ン一-龯]")
    for notebook in QUICKREF_NOTEBOOKS:
        payload = _load(Path("notebooks") / notebook)
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            raw = cell.get("source", [])
            source = "".join(raw) if isinstance(raw, list) else str(raw)
            assert not jp_pattern.search(source), f"{notebook}: {source}"
