from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.notebook_e2e

TARGET_UCS = {f"UC-{i}" for i in range(1, 9)} | {"UC-11", "UC-12"}


def test_phase26_3_manifest_entries_and_outputs_exist() -> None:
    payload = json.loads(Path("notebooks/phase26_3_execution_manifest.json").read_text("utf-8"))
    entries = payload.get("entries", [])
    assert isinstance(entries, list)
    by_uc = {entry.get("uc"): entry for entry in entries}
    assert TARGET_UCS.issubset(set(by_uc))

    for uc in TARGET_UCS:
        entry = by_uc[uc]
        assert entry.get("status") == "passed", uc
        notebook = Path(str(entry.get("notebook", "")))
        assert notebook.exists(), uc
        assert entry.get("artifact_path"), uc
        assert entry.get("metrics"), uc
        outputs = entry.get("outputs", [])
        assert outputs
        assert len(outputs) >= 3
        for out in outputs:
            assert Path(str(out)).exists(), out
