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
        "store-studio-predict-job",
        "store-studio-predict-result",
        "store-studio-hub-page",
        "store-studio-hub-total",
        "studio-mode-radio",
        "studio-model-hub-btn",
        "studio-guided-link",
        "studio-model-hub-offcanvas",
        "studio-hub-table",
        "studio-hub-load-btn",
        "studio-hub-delete-btn",
        "studio-hub-delete-confirm",
        "studio-train-upload",
        "studio-train-target-col",
        "studio-train-task-type",
        "studio-run-btn",
        "studio-run-status",
        "studio-run-progress",
        "studio-run-kpi",
        "studio-run-poll-interval",
        "studio-predict-poll-interval",
        "studio-predict-csv-download",
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


def test_studio_model_hub_table_and_load(monkeypatch) -> None:
    item = SimpleNamespace(
        path="artifacts/run1",
        run_id="run1",
        task_type="regression",
        created_at_utc="2026-02-19T00:00:00+00:00",
    )
    monkeypatch.setattr(
        app_module,
        "list_artifacts_page",
        lambda **_kw: SimpleNamespace(items=[item], total_count=1),
    )

    opened, rows, page, total, info, feedback = app_module._cb_studio_model_hub_table(
        1, 0, 0, 0, 0, False, 0, [], []
    )
    assert opened is True
    assert len(rows) == 1
    assert page == 0
    assert total == 1
    assert "1-1 / 1" in info
    assert feedback == ""

    monkeypatch.setattr(
        app_module,
        "get_artifact_spec",
        lambda _path: SimpleNamespace(
            artifact_path="artifacts/run1",
            run_id="run1",
            task_type="regression",
            target_col="y",
            feature_names=["x1", "x2"],
            feature_dtypes={},
            train_metrics={"rmse": 0.1},
            created_at_utc="2026-02-19T00:00:00+00:00",
        ),
    )
    store, mode_store, mode_radio, msg = app_module._cb_studio_load_artifact(
        1, [0], [{"path": "artifacts/run1", "run_id": "run1"}]
    )
    assert store["artifact_path"] == "artifacts/run1"
    assert mode_store == "inference"
    assert mode_radio == "inference"
    assert "[INFO]" in msg


def test_studio_delete_request_and_confirm(monkeypatch) -> None:
    displayed, message = app_module._cb_studio_request_delete(1, [0], [{"path": "artifacts/run1"}])
    assert displayed is True
    assert "cannot be undone" in message

    called: dict[str, str] = {}

    def _fake_delete(path: str) -> str:
        called["path"] = path
        return path

    monkeypatch.setattr(app_module, "delete_artifact_dir", _fake_delete)
    item = SimpleNamespace(
        path="artifacts/run2",
        run_id="run2",
        task_type="binary",
        created_at_utc="2026-02-19T00:00:00+00:00",
    )
    monkeypatch.setattr(
        app_module,
        "list_artifacts_page",
        lambda **_kw: SimpleNamespace(items=[item], total_count=1),
    )

    _opened, _rows, _page, _total, _info, feedback = app_module._cb_studio_model_hub_table(
        0,
        0,
        0,
        0,
        1,
        True,
        0,
        [0],
        [{"path": "artifacts/run1", "run_id": "run1"}],
    )
    assert called["path"] == "artifacts/run1"
    assert "Deleted artifact" in feedback


def test_studio_upload_predict_and_queue(monkeypatch) -> None:
    csv_content = "x1,x2,y\n1,2,0\n3,4,1\n"
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
                "columns": ["x1", "x2", "y"],
                "column_profiles": [
                    {"name": "x1", "dtype": "float64"},
                    {"name": "x2", "dtype": "float64"},
                    {"name": "y", "dtype": "int64"},
                ],
            },
        },
    )
    monkeypatch.setattr(
        app_module,
        "validate_prediction_data",
        lambda _spec, _path: [app_module.GuardRailResult("ok", "ok")],
    )

    data, msg, options, value, guardrails = app_module._cb_studio_upload_predict(
        contents,
        "pred.csv",
        {
            "artifact_path": "artifacts/run1",
            "run_id": "run1",
            "task_type": "binary",
            "target_col": "y",
            "feature_names": ["x1", "x2"],
            "feature_dtypes": {"x1": "float64", "x2": "float64"},
            "train_metrics": {},
            "created_at_utc": "2026-02-19T00:00:00+00:00",
        },
        {},
    )
    assert data["n_rows"] == 2
    assert "pred.csv" in msg
    assert len(options) == 4
    assert value == "y"
    assert guardrails is not None

    captured: dict[str, str] = {}

    def _fake_submit(invocation: RunInvocation):
        captured["action"] = invocation.action
        captured["artifact_path"] = str(invocation.artifact_path)
        captured["data_path"] = str(invocation.data_path)
        return SimpleNamespace(job_id="pred-job-1", status="queued", message="queued")

    monkeypatch.setattr(app_module, "submit_run_job", _fake_submit)
    log, state, disabled, n_intervals, status = app_module._cb_studio_predict(
        1,
        {
            "artifact_path": "artifacts/run1",
            "run_id": "run1",
            "task_type": "binary",
            "target_col": "y",
            "feature_names": ["x1", "x2"],
            "feature_dtypes": {},
            "train_metrics": {},
            "created_at_utc": "2026-02-19T00:00:00+00:00",
        },
        {"file_path": "/tmp/predict.csv"},
        "y",
    )
    assert "[QUEUED]" in log
    assert state["predict_job_id"] == "pred-job-1"
    assert captured["action"] == "predict"
    assert disabled is False
    assert n_intervals == 0
    assert status == "QUEUED"


def test_studio_poll_predict_success_and_auto_eval(monkeypatch, tmp_path) -> None:
    pred_job = GuiJobRecord(
        job_id="pred-job",
        status="succeeded",
        action="predict",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="predict"),
        result=GuiRunResult(
            success=True,
            message="ok",
            payload={
                "result": {
                    "prediction_csv_path": str(tmp_path / "pred.csv"),
                    "preview_rows": [{"p": 0.1}, {"p": 0.9}],
                    "total_count": 2,
                    "task_type": "binary",
                }
            },
        ),
    )
    eval_job = GuiJobRecord(
        job_id="eval-job",
        status="succeeded",
        action="evaluate",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="evaluate"),
        result=GuiRunResult(
            success=True,
            message="ok",
            payload={"metrics": {"auc": 0.88, "logloss": 0.3}},
        ),
    )
    (tmp_path / "pred.csv").write_text("p\n0.1\n0.9\n", encoding="utf-8")
    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: (lambda _p: SimpleNamespace(columns=["y"])),
    )

    jobs = {"pred-job": pred_job, "eval-job": eval_job}
    monkeypatch.setattr(app_module, "get_run_job", lambda job_id: jobs.get(job_id))
    monkeypatch.setattr(app_module, "list_run_job_logs", lambda _job_id, limit=120: [])
    monkeypatch.setattr(
        app_module,
        "submit_run_job",
        lambda inv: SimpleNamespace(
            job_id="eval-job",
            status="queued",
            message=f"queued {inv.action}",
        ),
    )

    out = app_module._cb_studio_poll_predict(
        1,
        {
            "predict_job_id": "pred-job",
            "stage": "predict",
            "status": "queued",
            "artifact_path": "artifacts/run1",
            "data_path": str(tmp_path / "input.csv"),
            "label_col": "y",
            "target_col": "y",
            "eval_requested": True,
            "eval_job_id": "",
            "eval_status": "",
        },
        {},
    )
    assert out[2] == "EVAL_QUEUED"
    assert out[3]["stage"] == "evaluate"
    assert out[5]["total_count"] == 2
    assert len(out[6]) == 2

    out_eval = app_module._cb_studio_poll_predict(
        2,
        {
            "predict_job_id": "pred-job",
            "stage": "evaluate",
            "status": "succeeded",
            "artifact_path": "artifacts/run1",
            "data_path": str(tmp_path / "input.csv"),
            "label_col": "y",
            "target_col": "y",
            "eval_requested": True,
            "eval_job_id": "eval-job",
            "eval_status": "queued",
        },
        out[5],
    )
    assert out_eval[2] == "EVAL_SUCCEEDED"
    assert out_eval[4] is True
    assert out_eval[9] is not None


def test_studio_download_predict_csv(monkeypatch, tmp_path) -> None:
    csv_path = tmp_path / "predict.csv"
    csv_path.write_text("prediction\n1.0\n", encoding="utf-8")
    payload = {"prediction_csv_path": str(csv_path)}
    res = app_module._cb_studio_download_predict_csv(1, payload)
    assert isinstance(res, dict)

    assert app_module._cb_studio_download_predict_csv(1, {}) is dash.no_update
