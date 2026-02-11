from __future__ import annotations

import json
from pathlib import Path


def _load_notebook() -> dict:
    path = Path("notebooks/lalonde_dr_analysis_workflow.ipynb")
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


def test_lalonde_notebook_has_required_dr_sections() -> None:
    nb = _load_notebook()
    source = _cells_source(nb)

    required_snippets = [
        "estimate_dr(",
        '"estimand": "att"',
        '"propensity_calibration": "platt"',
        "Naive vs IPW vs DR (ATT)",
        "e_raw",
        "e_hat",
        "smd_unweighted",
        "smd_weighted",
    ]
    for snippet in required_snippets:
        assert snippet in source, f"Missing notebook section: {snippet}"

