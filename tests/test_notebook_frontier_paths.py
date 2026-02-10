from __future__ import annotations

import json
from pathlib import Path


def _notebook_source() -> str:
    path = Path("notebooks/frontier_analysis_workflow.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            chunks.append("".join(src))
        else:
            chunks.append(str(src))
    return "\n".join(chunks)


def test_frontier_notebook_generates_data_internally() -> None:
    source = _notebook_source()
    assert "generate_frontier_base_data(" in source
    assert "generate_frontier_drifted_data(" in source


def test_frontier_notebook_uses_generated_model_parquet_for_fit() -> None:
    source = _notebook_source()
    assert 'TRAIN_PATH = OUT_DIR / "frontier_train.parquet"' in source
    assert '"data": {"path": str(TRAIN_PATH), "target": TARGET_COL}' in source
    assert '"task": {"type": "frontier"}' in source
