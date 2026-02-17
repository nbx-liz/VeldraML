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


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path) -> str:
    payload = _load(path)
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        src = cell.get("source", [])
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def test_phase26_3_notebook_contract() -> None:
    for nb in NOTEBOOKS:
        path = Path("notebooks") / nb
        payload = _load(path)
        src = _source(path)
        assert "## Overview" in src
        assert "## Learn More" in src
        assert "## Setup" in src
        assert "## Config Notes" in src
        assert "## Workflow" in src
        assert "### Output Annotation" in src
        assert "## Result Summary" in src
        assert "SUMMARY =" in src
        assert "matplotlib.use('Agg')" in src
        assert "from veldra.diagnostics import" in src
        assert "metrics_df" in src
        assert "display(" in src
        assert "placeholder" not in src.lower()

        code_cells = [cell for cell in payload.get("cells", []) if cell.get("cell_type") == "code"]
        assert code_cells, nb
        assert all(cell.get("execution_count") is not None for cell in code_cells), nb
        assert any(cell.get("outputs") for cell in code_cells), nb
