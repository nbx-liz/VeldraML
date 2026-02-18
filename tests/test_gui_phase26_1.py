from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from types import SimpleNamespace

import dash
import pytest

from veldra.gui import app as app_module
from veldra.gui import services as services_module
from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def _job(
    *,
    job_id: str,
    status: str,
    created_at_utc: str = "2026-02-16T00:00:00+00:00",
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
    output_path: str | None = None,
    error_message: str | None = None,
) -> GuiJobRecord:
    payload = {"artifact_path": f"artifacts/{job_id}"}
    if output_path:
        payload["output_path"] = output_path
    result = GuiRunResult(success=status == "succeeded", message="ok", payload=payload)
    return GuiJobRecord(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        action="export_html_report",
        created_at_utc=created_at_utc,
        updated_at_utc=created_at_utc,
        invocation=RunInvocation(action="export_html_report", artifact_path=f"artifacts/{job_id}"),
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        result=result,
        error_message=error_message,
    )


def test_runs_callbacks_render_jst_timestamps(monkeypatch) -> None:
    job = _job(
        job_id="run-1",
        status="succeeded",
        created_at_utc="2026-02-16T00:00:00+00:00",
        started_at_utc="2026-02-16T00:01:00+00:00",
        finished_at_utc="2026-02-16T00:02:00+00:00",
    )
    monkeypatch.setattr(app_module, "list_run_jobs_filtered", lambda **_kwargs: [job])
    rows = app_module._cb_refresh_runs_table(1, "/runs", "", "", "")
    assert rows[0]["created_at_utc"] == "2026-02-16 09:00:00 JST"
    assert rows[0]["started_at_utc"] == "2026-02-16 09:01:00 JST"
    assert rows[0]["finished_at_utc"] == "2026-02-16 09:02:00 JST"


def test_show_selected_job_detail_uses_n_a_for_missing_timestamps(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "get_run_job",
        lambda _job_id: _job(job_id="run-2", status="queued"),
    )
    monkeypatch.setattr(app_module, "list_run_job_logs", lambda _job_id, limit=200: [])
    detail, _, _, _, _ = app_module._cb_show_selected_job_detail(
        [0], 0, 0, [{"job_id": "run-2"}], None, 200
    )
    text = str(detail)
    assert text.count("n/a") >= 2


def test_export_job_poll_success_triggers_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "result-export-html-btn.n_clicks"}]),
    )
    monkeypatch.setattr(
        app_module,
        "submit_run_job",
        lambda _inv: SimpleNamespace(job_id="job-export-1"),
    )
    status, state, disabled, n_intervals = app_module._cb_result_export_actions(
        0, 1, "artifacts/run-1"
    )
    assert "生成中..." in status
    assert state == {"job_id": "job-export-1", "action": "export_html_report"}
    assert disabled is False
    assert n_intervals == 0

    report = tmp_path / "report.html"
    report.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(
        app_module,
        "get_run_job",
        lambda _job_id: _job(job_id="job-export-1", status="succeeded", output_path=str(report)),
    )
    download, final_status, next_state, poll_disabled = app_module._cb_poll_result_export_job(
        1, state
    )
    assert isinstance(download, dict)
    assert final_status == "ダウンロード完了"
    assert next_state is None
    assert poll_disabled is True


def test_export_job_poll_failure_does_not_trigger_download(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "get_run_job",
        lambda _job_id: _job(job_id="job-export-2", status="failed", error_message="boom"),
    )
    download, status, state, poll_disabled = app_module._cb_poll_result_export_job(
        1, {"job_id": "job-export-2", "action": "export_html_report"}
    )
    assert download is dash.no_update
    assert "[ERROR]" in status
    assert "boom" in status
    assert state is None
    assert poll_disabled is True


def test_update_result_extras_prefers_artifact_training_history(monkeypatch) -> None:
    seen_histories: list[dict] = []

    def _capture(history: dict) -> dict:
        seen_histories.append(history)
        return {"history": history}

    monkeypatch.setattr(app_module, "plot_learning_curves", _capture)

    class _ArtifactWithHistory:
        training_history = {"folds": [{"fold": 0}]}
        config = {"task": {"type": "regression"}}
        run_config = {"task": {"type": "regression"}}
        feature_schema = {"feature_names": ["x1"], "n_rows": 1}
        task_type = "regression"

    class _ArtifactWithoutHistory:
        training_history = None
        config = {"task": {"type": "causal"}}
        run_config = {"task": {"type": "causal"}}
        feature_schema = {"feature_names": [], "n_rows": 1}
        task_type = "causal"

    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(
            load=lambda path: _ArtifactWithHistory()
            if "with-history" in str(path)
            else _ArtifactWithoutHistory()
        ),
    )

    app_module._cb_update_result_extras("with-history")
    app_module._cb_update_result_extras("without-history")

    assert seen_histories[0] == {"folds": [{"fold": 0}]}
    assert seen_histories[1] == {}


def test_export_output_path_uses_jst_timestamp(monkeypatch, tmp_path) -> None:
    fixed_utc = datetime(2026, 2, 16, 0, 0, 0, tzinfo=UTC)

    class _DatetimeStub:
        @staticmethod
        def now(_tz):
            return fixed_utc

    monkeypatch.setattr(services_module, "datetime", _DatetimeStub)
    out = services_module._export_output_path(str(tmp_path / "artifact"), "report")
    assert out.name.startswith("report_20260216_090000")
