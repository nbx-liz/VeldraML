from __future__ import annotations

from types import SimpleNamespace

import dash

from veldra.gui import app as app_module
from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation


def _job(
    job_id: str, artifact_path: str = "artifacts/a", config_path: str | None = "cfg.yaml"
) -> GuiJobRecord:
    return GuiJobRecord(
        job_id=job_id,
        status="succeeded",
        action="fit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(
            action="fit", artifact_path=artifact_path, config_path=config_path
        ),
        result=GuiRunResult(success=True, message="ok", payload={"artifact_path": artifact_path}),
    )


def test_runs_actions_table_selection_branches(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-table.active_cell"}]),
    )

    state, path, msg = app_module._cb_runs_actions(0, 0, 0, 0, 0, None, [], {}, None)
    assert path is dash.no_update
    assert "Select a run row" in msg

    state2, _, msg2 = app_module._cb_runs_actions(
        0,
        0,
        0,
        0,
        0,
        {"row": 9, "column_id": "x"},
        [],
        {},
        [{"artifact_path": "artifacts/a"}],
    )
    assert state2 == app_module._ensure_workflow_state_defaults({})
    assert "Invalid run row" in msg2

    _, _, msg3 = app_module._cb_runs_actions(
        0,
        0,
        0,
        0,
        0,
        {"row": 0, "column_id": "x"},
        [],
        {},
        [{"artifact_path": ""}],
    )
    assert "No artifact path" in msg3

    _, _, msg4 = app_module._cb_runs_actions(
        0,
        0,
        0,
        0,
        0,
        {"row": 0, "column_id": "x"},
        [],
        {},
        [{"artifact_path": "artifacts/a"}],
    )
    assert "No shortcut action" in msg4


def test_runs_actions_button_branches(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-delete-btn.n_clicks"}]),
    )

    _, _, msg_select = app_module._cb_runs_actions(0, 0, 0, 0, 0, None, [], {}, [])
    assert "Select one or more runs" in msg_select

    monkeypatch.setattr(app_module, "delete_run_jobs", lambda _ids: 2)
    _, _, msg_delete = app_module._cb_runs_actions(0, 0, 1, 0, 0, None, ["j1", "j2"], {}, [])
    assert "Deleted 2" in msg_delete

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-compare-btn.n_clicks"}]),
    )
    _, _, msg_cmp_count = app_module._cb_runs_actions(1, 0, 0, 0, 0, None, ["j1"], {}, [])
    assert "exactly two" in msg_cmp_count

    monkeypatch.setattr(app_module, "get_run_job", lambda _id: None)
    _, _, msg_cmp_missing = app_module._cb_runs_actions(1, 0, 0, 0, 0, None, ["j1", "j2"], {}, [])
    assert "Artifacts not found" in msg_cmp_missing

    monkeypatch.setattr(app_module, "get_run_job", lambda _id: None)
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-clone-btn.n_clicks"}]),
    )
    _, _, msg_primary_none = app_module._cb_runs_actions(0, 1, 0, 0, 0, None, ["j1"], {}, [])
    assert "Run not found" in msg_primary_none


def test_runs_actions_clone_view_migrate_and_error(monkeypatch) -> None:
    primary = _job("j1")
    monkeypatch.setattr(app_module, "get_run_job", lambda _id: primary)

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-clone-btn.n_clicks"}]),
    )
    monkeypatch.setattr(app_module, "load_job_config_yaml", lambda _job: "- a\n- b\n")
    _, _, msg_clone_invalid = app_module._cb_runs_actions(0, 1, 0, 0, 0, None, ["j1"], {}, [])
    assert "Invalid config payload" in msg_clone_invalid

    monkeypatch.setattr(
        app_module, "load_job_config_yaml", lambda _job: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    _, _, msg_clone_error = app_module._cb_runs_actions(0, 1, 0, 0, 0, None, ["j1"], {}, [])
    assert "Run action failed" in msg_clone_error

    no_path_job = _job("j2", artifact_path="")
    monkeypatch.setattr(app_module, "get_run_job", lambda _id: no_path_job)
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-view-results-btn.n_clicks"}]),
    )
    _, _, msg_view = app_module._cb_runs_actions(0, 0, 0, 1, 0, None, ["j2"], {}, [])
    assert "No artifact path" in msg_view

    no_cfg_job = _job("j3", config_path=None)
    monkeypatch.setattr(app_module, "get_run_job", lambda _id: no_cfg_job)
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-migrate-btn.n_clicks"}]),
    )
    _, _, msg_migrate = app_module._cb_runs_actions(0, 0, 0, 0, 1, None, ["j3"], {}, [])
    assert "No config path" in msg_migrate

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "runs-unknown-btn.n_clicks"}]),
    )
    _, _, msg_no_action = app_module._cb_runs_actions(0, 0, 0, 0, 0, None, ["j1"], {}, [])
    assert "No action" in msg_no_action


def test_populate_compare_options_and_compare_runs_error_paths(monkeypatch) -> None:
    assert app_module._cb_populate_compare_options("/run", {}) == ([], [], None, None)

    msg, rows, fig, diff = app_module._cb_compare_runs(None, "b")
    assert "Select at least 2 artifacts" in msg
    assert rows == []
    assert hasattr(fig, "to_dict")
    assert diff == ""

    monkeypatch.setattr(
        app_module,
        "compare_artifacts_multi",
        lambda _a, _b: (_ for _ in ()).throw(RuntimeError("compare boom")),
    )
    msg2, rows2, fig2, diff2 = app_module._cb_compare_runs("a", "b")
    assert "Compare failed" in msg2
    assert rows2 == []
    assert hasattr(fig2, "to_dict")
    assert diff2 == ""
