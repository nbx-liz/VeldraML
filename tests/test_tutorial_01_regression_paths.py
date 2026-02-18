from __future__ import annotations

import json
from pathlib import Path


def _notebook_source() -> str:
    path = Path("notebooks/tutorials/tutorial_01_regression_basics.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            chunks.append("".join(src))
        else:
            chunks.append(str(src))
    return "\n".join(chunks)


def test_notebook_generates_data_internally() -> None:
    source = _notebook_source()
    assert "generate_saas_ltv_data(" in source
    assert "generate_drifted_data(" in source


def test_notebook_uses_generated_model_file_for_fit() -> None:
    source = _notebook_source()
    assert 'TRAIN_MODEL_PATH = OUT_DIR / "train_model_prepared.parquet"' in source
    assert "to_parquet(" in source
    assert '"data": {"path": str(TRAIN_MODEL_PATH), "target": TARGET_COL}' in source
