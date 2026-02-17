from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = [
    "phase26_2_uc01_regression_fit_evaluate.ipynb",
    "phase26_2_uc02_binary_tune_evaluate.ipynb",
    "phase26_2_uc03_frontier_fit_evaluate.ipynb",
    "phase26_2_uc04_causal_dr_estimate.ipynb",
    "phase26_2_uc05_causal_drdid_estimate.ipynb",
    "phase26_2_uc06_causal_dr_tune.ipynb",
    "phase26_2_uc07_artifact_evaluate.ipynb",
    "phase26_2_uc08_artifact_reevaluate.ipynb",
    "phase26_3_uc_multiclass_fit_evaluate.ipynb",
    "phase26_3_uc_timeseries_fit_evaluate.ipynb",
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
        assert "## Setup" in src
        assert "## Workflow" in src
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
