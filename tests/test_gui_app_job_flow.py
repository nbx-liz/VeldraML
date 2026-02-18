from __future__ import annotations

from types import SimpleNamespace

import dash

from veldra.gui import app as app_module
from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation


def _job(
    job_id: str,
    status: str,
    action: str = "fit",
    payload: dict | None = None,
    error_message: str | None = None,
) -> GuiJobRecord:
    result = GuiRunResult(True, "ok", payload or {}) if payload is not None else None
    return GuiJobRecord(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        action=action,
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action=action),
        result=result,
        error_message=error_message,
    )


def test_enqueue_run_job_success_and_error(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_ensure_default_run_config", lambda p: p or "cfg.yml")
    monkeypatch.setattr(
        app_module,
        "submit_run_job",
        lambda inv: SimpleNamespace(job_id="j1", message=f"queued {inv.action}"),
    )
    msg = app_module._cb_enqueue_run_job(
        1, "fit", "a: 1", "cfg.yml", "data.csv", "", "", "python", "high"
    )
    assert "[QUEUED]" in msg

    monkeypatch.setattr(
        app_module,
        "submit_run_job",
        lambda inv: (_ for _ in ()).throw(RuntimeError("enqueue fail")),
    )
    msg2 = app_module._cb_enqueue_run_job(
        1, "fit", "a: 1", "cfg.yml", "data.csv", "", "", "python", "normal"
    )
    assert "[ERROR]" in msg2
    assert "enqueue fail" in msg2


def test_refresh_run_jobs_transition_and_redirect(monkeypatch) -> None:
    jobs = [
        _job("j1", "succeeded", payload={"artifact_path": "artifacts/a"}),
        _job("j2", "failed"),
    ]
    monkeypatch.setattr(app_module, "list_run_jobs", lambda limit=100: jobs)
    monkeypatch.setattr(app_module, "make_toast", lambda message, icon: f"{icon}:{message}")

    table, toast, status, state, next_path = app_module._cb_refresh_run_jobs(
        1,
        0,
        {"j1": "running", "j2": "queued"},
        {},
        "/run",
        [],
    )
    assert table is not None
    assert status["j1"] == "succeeded"
    assert "Task fit" in toast
    assert state["last_run_artifact"] == "artifacts/a"
    assert next_path == "/results"

    _, _, _, _, next_batch = app_module._cb_refresh_run_jobs(
        1,
        0,
        {"j1": "running"},
        {},
        "/run",
        ["enabled"],
    )
    assert next_batch is dash.no_update


def test_show_selected_job_detail_branches(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "list_run_job_logs", lambda _job_id, limit=200: [])
    assert app_module._cb_show_selected_job_detail(None, 0, 0, None, None, 200)[0].startswith(
        "Select"
    )
    assert "not found" in app_module._cb_show_selected_job_detail(
        [9], 0, 0, [{"job_id": "j1"}], None, 200
    )[0]
    assert app_module._cb_show_selected_job_detail([0], 0, 0, [{"job_id": "j1"}], None, 200)[
        1
    ] is True

    monkeypatch.setattr(app_module, "get_run_job", lambda _jid: None)
    assert "unavailable" in app_module._cb_show_selected_job_detail(
        [0], 0, 0, [{"job_id": "j1"}], None, 200
    )[0]

    job = _job("j1", "failed", error_message="boom")
    monkeypatch.setattr(app_module, "get_run_job", lambda _jid: job)
    detail = app_module._cb_show_selected_job_detail([0], 0, 0, [{"job_id": "j1"}], None, 200)
    assert "FAILED" in str(detail[0])
    assert "boom" in str(detail[0])
    assert detail[1] is True

    running_job = _job("j2", "running")
    monkeypatch.setattr(app_module, "get_run_job", lambda _jid: running_job)
    detail2 = app_module._cb_show_selected_job_detail([0], 0, 0, [{"job_id": "j2"}], None, 200)
    assert detail2[1] is False


def test_cancel_job_result(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "cancel_run_job",
        lambda _job_id: SimpleNamespace(message="canceled"),
    )
    assert app_module._cb_cancel_job(1, "j1") == "[INFO] canceled"
    assert app_module._cb_cancel_job(1, None) == ""

    monkeypatch.setattr(
        app_module,
        "cancel_run_job",
        lambda _job_id: (_ for _ in ()).throw(RuntimeError("cancel fail")),
    )
    out = app_module._cb_cancel_job(1, "j1")
    assert out.startswith("[ERROR]")


def test_set_job_priority_result(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "set_run_job_priority",
        lambda _job_id, _priority: SimpleNamespace(message="priority updated"),
    )
    assert app_module._cb_set_job_priority(1, "j1", "high") == "[INFO] priority updated"
    assert app_module._cb_set_job_priority(1, None, "high").startswith("[ERROR]")
