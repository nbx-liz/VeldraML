from __future__ import annotations

from types import SimpleNamespace

import dash
import pandas as pd

from veldra.gui import app as app_module
from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation


def _job(status: str, payload: dict | None = None, error: str | None = None) -> GuiJobRecord:
    return GuiJobRecord(
        job_id="job-x",
        status=status,
        action="export_html_report",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="export_html_report", artifact_path="artifacts/r1"),
        result=GuiRunResult(success=status == "succeeded", message="m", payload=payload or {}),
        error_message=error,
    )


def test_update_result_extras_error_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(load=lambda _p: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    curve, fold, causal, opts, drill, cfg, summary = app_module._cb_update_result_extras(
        "artifacts/x"
    )
    assert hasattr(curve, "to_dict")
    assert hasattr(fold, "to_dict")
    assert hasattr(causal, "to_dict")
    assert opts == []
    assert hasattr(drill, "to_dict")
    assert "Error" in cfg
    assert "Error" in summary


def test_result_export_help_variants() -> None:
    assert "Excel" in app_module._cb_result_export_help()
    assert "Excel" in app_module._cb_result_export_help_for_artifact("artifacts/a")


def test_result_eval_precheck_branches(monkeypatch) -> None:
    assert "Select an artifact" in str(app_module._cb_result_eval_precheck(None, "data.csv"))
    assert "Evaluation Data Path" in str(app_module._cb_result_eval_precheck("artifacts/a", ""))

    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(load=lambda _p: SimpleNamespace(feature_schema={})),
    )
    monkeypatch.setattr(
        app_module, "_get_load_tabular_data", lambda: lambda _p: pd.DataFrame({"x": [1]})
    )
    no_schema = app_module._cb_result_eval_precheck("artifacts/a", "data.csv")
    assert "not available" in str(no_schema)

    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(
            load=lambda _p: SimpleNamespace(feature_schema={"feature_names": ["a", "b"]})
        ),
    )
    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: lambda _p: pd.DataFrame({"a": [1], "c": [2]}),
    )
    mismatch = app_module._cb_result_eval_precheck("artifacts/a", "data.csv")
    text = str(mismatch)
    assert "Missing features" in text
    assert "Extra columns" in text

    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(load=lambda _p: (_ for _ in ()).throw(RuntimeError("precheck failed"))),
    )
    fail = app_module._cb_result_eval_precheck("artifacts/a", "data.csv")
    assert "Precheck failed" in str(fail)


def test_result_shortcut_highlight_export_and_non_results() -> None:
    assert app_module._cb_result_shortcut_highlight(
        {"results_shortcut_focus": "export"}, "/results"
    )[1].endswith("border border-warning")
    non_results = app_module._cb_result_shortcut_highlight({}, "/run")
    assert non_results == (
        "w-100 mb-3",
        "me-2 result-export-btn",
        "me-2 result-export-btn",
        "me-2 result-export-btn",
    )


def test_result_export_actions_and_polling_branches(monkeypatch, tmp_path) -> None:
    no_art = app_module._cb_result_export_actions(1, 0, 0, None)
    assert "Select an artifact" in no_art[0]

    monkeypatch.setattr(app_module, "submit_run_job", lambda _inv: SimpleNamespace(job_id="job-1"))

    class _NoTriggered:
        @property
        def triggered(self):
            return []

    monkeypatch.setattr(app_module, "callback_context", _NoTriggered())
    started = app_module._cb_result_export_actions(1, 0, 0, "artifacts/a")
    assert "生成中" in started[0]
    assert started[1] == {"job_id": "job-1", "action": "export_excel"}

    monkeypatch.setattr(
        app_module,
        "submit_run_job",
        lambda _inv: (_ for _ in ()).throw(RuntimeError("submit failed")),
    )
    failed = app_module._cb_result_export_actions(1, 0, 0, "artifacts/a")
    assert failed[0].startswith("[ERROR]")

    assert app_module._cb_poll_result_export_job(1, None) == (
        dash.no_update,
        dash.no_update,
        dash.no_update,
        True,
    )

    missing_id = app_module._cb_poll_result_export_job(1, {"action": "export_excel"})
    assert "job id is missing" in missing_id[1]

    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: None)
    not_found = app_module._cb_poll_result_export_job(1, {"job_id": "j1", "action": "export_excel"})
    assert "not found" in not_found[1]

    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: _job("running"))
    running = app_module._cb_poll_result_export_job(1, {"job_id": "j1", "action": "export_excel"})
    assert "生成中" in running[1]

    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: _job("succeeded", payload={}))
    no_path = app_module._cb_poll_result_export_job(1, {"job_id": "j1", "action": "export_excel"})
    assert "output path is missing" in no_path[1]

    monkeypatch.setattr(
        app_module,
        "get_run_job",
        lambda _job_id: _job("succeeded", payload={"output_path": str(tmp_path / "none.xlsx")}),
    )
    missing_file = app_module._cb_poll_result_export_job(
        1, {"job_id": "j1", "action": "export_excel"}
    )
    assert "file not found" in missing_file[1]

    file_path = tmp_path / "ok.xlsx"
    file_path.write_text("ok", encoding="utf-8")
    monkeypatch.setattr(
        app_module,
        "get_run_job",
        lambda _job_id: _job("succeeded", payload={"output_path": str(file_path)}),
    )
    done = app_module._cb_poll_result_export_job(1, {"job_id": "j1", "action": "export_excel"})
    assert isinstance(done[0], dict)
    assert done[1] == "ダウンロード完了"

    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: _job("failed", error="boom"))
    failed_job = app_module._cb_poll_result_export_job(
        1, {"job_id": "j1", "action": "export_excel"}
    )
    assert failed_job[1].startswith("[ERROR]")

    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: _job("queued"))
    queued = app_module._cb_poll_result_export_job(1, {"job_id": "j1", "action": "export_excel"})
    assert "生成中" in queued[1]

    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: _job("mystery"))
    unknown = app_module._cb_poll_result_export_job(1, {"job_id": "j1", "action": "export_excel"})
    assert "Export status" in unknown[1]


def test_result_download_config_error_branch(monkeypatch) -> None:
    assert app_module._cb_result_download_config(1, None) is dash.no_update

    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(load=lambda _p: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    assert app_module._cb_result_download_config(1, "artifacts/x") is dash.no_update
