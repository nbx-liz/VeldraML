from __future__ import annotations

import json
from pathlib import Path


def _notebook_source() -> str:
    path = Path("notebooks/tutorials/tutorial_06_causal_drdid_lalonde.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            chunks.append("".join(src))
        else:
            chunks.append(str(src))
    return "\n".join(chunks)


def test_lalonde_drdid_notebook_uses_url_with_local_cache() -> None:
    source = _notebook_source()
    assert 'OUT_DIR = ROOT / "examples" / "out" / "notebook_lalonde_drdid"' in source
    assert 'CACHE_PATH = OUT_DIR / "lalonde_raw.parquet"' in source
    assert 'PANEL_PATH = OUT_DIR / "lalonde_panel.parquet"' in source
    assert (
        'LALONDE_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv"'
        in source
    )
    assert "FALLBACK_CACHE_PATHS = [" in source
    assert "if candidate.exists():" in source
    assert "lalonde_df.to_parquet(CACHE_PATH, index=False)" in source
    assert "panel_df.to_parquet(PANEL_PATH, index=False)" in source


def test_lalonde_drdid_notebook_uses_panel_cache_for_runconfig() -> None:
    source = _notebook_source()
    assert '"path": str(PANEL_PATH)' in source
    assert '"target": TARGET_COL' in source
    assert '"task": {"type": "regression"}' in source
    assert '"method": "dr_did"' in source
    assert 'SUMMARY_PATH = OUT_DIR / "lalonde_drdid_summary.json"' in source
