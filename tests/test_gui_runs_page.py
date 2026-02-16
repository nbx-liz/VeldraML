from __future__ import annotations

import importlib.util

import pytest

from veldra.gui import app as app_module
from veldra.gui.pages import runs_page
from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def _collect_ids(component, out: set[str]) -> None:
    if component is None:
        return
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        out.add(cid)
    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            _collect_ids(child, out)
    else:
        _collect_ids(children, out)


def _job(job_id: str, status: str = "succeeded") -> GuiJobRecord:
    return GuiJobRecord(
        job_id=job_id,
        status=status,
        action="fit",
        created_at_utc="2026-02-16T00:00:00+00:00",
        updated_at_utc="2026-02-16T00:00:00+00:00",
        invocation=RunInvocation(action="fit", artifact_path=f"artifacts/{job_id}"),
        result=GuiRunResult(True, "ok", {"artifact_path": f"artifacts/{job_id}"}),
    )


def test_runs_layout_has_controls() -> None:
    layout = runs_page.layout()
    ids: set[str] = set()
    _collect_ids(layout, ids)
    assert "runs-table" in ids
    assert "runs-compare-btn" in ids
    assert "runs-delete-btn" in ids


def test_refresh_runs_table(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "list_run_jobs_filtered", lambda **_k: [_job("a1"), _job("b2")])
    rows = app_module._cb_refresh_runs_table(1, "/runs", "", "", "")
    assert len(rows) == 2
    assert rows[0]["job_id"] == "a1"
    assert rows[0]["created_at_utc"].endswith("JST")
    assert rows[0]["started_at_utc"] == "n/a"
    assert rows[0]["finished_at_utc"] == "n/a"


def test_runs_selection_detail(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_run_job", lambda job_id: _job(job_id))
    detail, selected = app_module._cb_runs_selection_detail([0], [{"job_id": "a1"}])
    assert "Job ID" in detail
    assert selected == ["a1"]
