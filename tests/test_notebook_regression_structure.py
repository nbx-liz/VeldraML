from __future__ import annotations

import json
from pathlib import Path


def _load_notebook() -> dict:
    path = Path("notebooks/regression_analysis_workflow.ipynb")
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


def test_notebook_has_required_analysis_sections() -> None:
    nb = _load_notebook()
    source = _cells_source(nb)

    required_snippets = [
        "build_pred_df(",
        "plot_actual_vs_pred(",
        "plot_error_distribution(",
        "compute_lgb_importance(",
        "compute_shap_summary(",
        "simulate(",
        "export(",
    ]
    for snippet in required_snippets:
        assert snippet in source, f"Missing notebook section: {snippet}"


def test_notebook_uses_lightgbm_pred_contrib_for_shap() -> None:
    nb = _load_notebook()
    source = _cells_source(nb)

    assert "pred_contrib=True" in source
    assert "compute_shap_summary(" in source
    assert "LightGBM pred_contrib" in source
