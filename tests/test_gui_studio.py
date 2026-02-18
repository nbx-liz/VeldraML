from __future__ import annotations

import base64
from types import SimpleNamespace

import dash
import pytest

from veldra.gui import app as app_module
from veldra.gui.pages import studio_page
from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation


def _collect_ids(component, out: set[str]) -> None:
    if component is None:
        return
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        out.add(component_id)
    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            _collect_ids(child, out)
    else:
        _collect_ids(children, out)


def test_studio_layout_has_required_ids() -> None:
    layout = studio_page.layout()
    ids: set[str] = set()
    _collect_ids(layout, ids)

    expected = {
        "store-studio-mode",
        "store-studio-train-data",
        "store-studio-predict-data",
        "store-studio-artifact",
        "store-studio-last-job",
        "store-studio-predict-result",
        "studio-mode-radio",
        "studio-model-hub-btn",
        "studio-guided-link",
        "studio-train-upload",
        "studio-train-target-col",
        "studio-train-task-type",
        "studio-run-btn",
        "studio-run-status",
        "studio-run-progress",
        "studio-run-kpi",
        "studio-run-poll-interval",
    }
    assert expected.issubset(ids)


def test_redirect_root_to_studio() -> None:
    assert app_module._cb_redirect_root_to_studio("/") == "/studio"
    assert app_module._cb_redirect_root_to_studio(None) == "/studio"
    assert app_module._cb_redirect_root_to_studio("/data") is dash.no_update


def test_build_studio_run_config_contract() -> None:
    yaml_text = app_module._build_studio_run_config(
        {"file_path": "data.csv", "target_col": "target", "task_type": "regression"},
        "target",
        "regression",
        "kfold",
        5,
        None,
        None,
        "expanding",
        None,
        0,
        0,
        0.05,
        31,
        -1,
        300,
        100,
        False,
        "standard",
        30,
        None,
    )
    cfg = app_module.validate_config(yaml_text)
    assert cfg.task.type == "regression"
    assert cfg.data.path == "data.csv"
    assert cfg.data.target == "target"

    yaml_ts = app_module._build_studio_run_config(
        {"file_path": "data.csv", "target_col": "y", "task_type": "binary"},
        "y",
        "binary",
        "timeseries",
        4,
        None,
        "event_time",
        "blocked",
        24,
        1,
        2,
        0.03,
        63,
        8,
        250,
        50,
        True,
        "fast",
        20,
        "auc",
    )
    cfg_ts = app_module.validate_config(yaml_ts)
    assert cfg_ts.task.type == "binary"
    assert cfg_ts.tuning.enabled is True
    assert cfg_ts.split.type == "timeseries"
    assert cfg_ts.split.time_col == "event_time"


@pytest.mark.parametrize(
    ("tune_enabled", "expected_action"),
    [
        (False, "fit"),
        (True, "tune"),
    ],
)
def test_studio_run_queues_job(monkeypatch, tune_enabled: bool, expected_action: str) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        app_module,
        "validate_config_with_guidance",
        lambda _yaml: {"ok": True, "errors": [], "warnings": [], "timestamp_utc": "now"},
    )

    def _fake_submit(invocation: RunInvocation):
        captured["action"] = invocation.action
        return SimpleNamespace(job_id="studio-job-1", status="queued", message="queued")

    monkeypatch.setattr(app_module, "submit_run_job", _fake_submit)

    msg, store, polling_disabled, n_intervals, status = app_module._cb_studio_run(
        1,
        {"file_path": "data.csv", "target_col": "y", "task_type": "regression"},
        "y",
        "regression",
        "kfold",
        5,
        None,
        None,
        "expanding",
        None,
        0,
        0,
        0.05,
        31,
        -1,
        300,
        100,
        tune_enabled,
        "standard",
        30,
        None,
    )

    assert "[QUEUED]" in msg
    assert store["job_id"] == "studio-job-1"
    assert captured["action"] == expected_action
    assert polling_disabled is False
    assert n_intervals == 0
    assert status == "QUEUED"


def test_studio_upload_train_populates_store(monkeypatch) -> None:
    csv_content = "a,b,target\n1,2,0\n3,4,1\n"
    encoded = base64.b64encode(csv_content.encode("utf-8")).decode("ascii")
    contents = f"data:text/csv;base64,{encoded}"

    monkeypatch.setattr(
        app_module,
        "inspect_data",
        lambda _path: {
            "success": True,
            "stats": {
                "n_rows": 2,
                "n_cols": 3,
                "columns": ["a", "b", "target"],
            },
        },
    )
    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: (lambda _path: SimpleNamespace(columns=["a", "b", "target"])),
    )
    monkeypatch.setattr(app_module, "infer_task_type", lambda _df, _target: "binary")

    store, msg, options, value, task = app_module._cb_studio_upload_train(contents, "train.csv", {})
    assert store["n_rows"] == 2
    assert store["target_col"] == "target"
    assert "train.csv" in msg
    assert len(options) == 3
    assert value == "target"
    assert task == "binary"


def test_studio_poll_updates_workflow_for_success(monkeypatch) -> None:
    job = GuiJobRecord(
        job_id="j1",
        status="succeeded",
        action="fit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="fit"),
        result=GuiRunResult(
            success=True,
            message="ok",
            payload={"result": {"artifact_path": "artifacts/run-1"}, "metrics": {"rmse": 0.12}},
        ),
    )
    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: job)
    monkeypatch.setattr(app_module, "list_run_job_logs", lambda _job_id, limit=120: [])

    progress, kpi, log_text, status, last_job, disabled, workflow = app_module._cb_studio_poll_job(
        1,
        {"job_id": "j1", "action": "fit", "status": "queued"},
        {},
    )

    assert progress is not None
    assert kpi is not None
    assert "SUCCEEDED" in log_text
    assert status == "SUCCEEDED"
    assert last_job["status"] == "succeeded"
    assert disabled is True
    assert workflow["last_job_succeeded"] is True
    assert workflow["last_run_artifact"] == "artifacts/run-1"


def test_studio_poll_running_keeps_interval_enabled(monkeypatch) -> None:
    job = GuiJobRecord(
        job_id="j2",
        status="running",
        action="tune",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="tune"),
    )
    monkeypatch.setattr(app_module, "get_run_job", lambda _job_id: job)
    monkeypatch.setattr(app_module, "list_run_job_logs", lambda _job_id, limit=120: [])

    _progress, _kpi, log_text, status, _last_job, disabled, _workflow = (
        app_module._cb_studio_poll_job(
            1,
            {"job_id": "j2"},
            {},
        )
    )
    assert "RUNNING" in log_text
    assert status == "RUNNING"
    assert disabled is False
