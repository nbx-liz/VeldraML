from __future__ import annotations

import json
from pathlib import Path


def _load_notebook() -> dict:
    path = Path("notebooks/simulate_analysis_workflow.ipynb")
    assert path.exists(), "Notebook file is missing."
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


def test_simulate_notebook_has_required_sections() -> None:
    nb = _load_notebook()
    source = _cells_source(nb)

    required_snippets = [
        "generate_saas_ltv_data(",
        "simulate(",
        "scenario_kpi",
        "segment_kpi",
        "top_impacted",
    ]
    for snippet in required_snippets:
        assert snippet in source, f"Missing notebook section: {snippet}"


def test_simulate_notebook_uses_internal_data_generation() -> None:
    nb = _load_notebook()
    source = _cells_source(nb)

    assert "TRAIN_PATH = OUT_DIR / \"simulate_train.parquet\"" in source
    assert "to_parquet(" in source
    assert "\"task\": {\"type\": \"regression\"}" in source
