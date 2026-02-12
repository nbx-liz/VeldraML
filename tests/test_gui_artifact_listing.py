from __future__ import annotations

import json

from veldra.gui.services import list_artifacts


def test_list_artifacts_returns_sorted_summaries(tmp_path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)

    older = root / "run_old"
    newer = root / "run_new"
    broken = root / "run_broken"
    for item in (older, newer, broken):
        item.mkdir(parents=True, exist_ok=True)

    (older / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "old",
                "task_type": "regression",
                "created_at_utc": "2026-02-11T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (newer / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "new",
                "task_type": "binary",
                "created_at_utc": "2026-02-11T11:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (broken / "manifest.json").write_text("not-json", encoding="utf-8")

    items = list_artifacts(str(root))
    assert [item.run_id for item in items] == ["new", "old", "run_broken"]
    assert items[2].task_type == "unknown"


def test_list_artifacts_handles_missing_directory(tmp_path) -> None:
    missing_root = tmp_path / "missing"
    assert list_artifacts(str(missing_root)) == []
