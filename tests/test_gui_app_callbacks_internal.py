from __future__ import annotations

import importlib.util
import json
import types

import pytest

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def _wrapped(app, key: str):
    return app.callback_map[key]["callback"].__wrapped__


def _wrapped_by_prefix(app, prefix: str):
    for key, value in app.callback_map.items():
        if key.startswith(prefix):
            return value["callback"].__wrapped__
    raise AssertionError(f"callback prefix not found: {prefix}")


def test_callback_wrappers_cover_branches(monkeypatch) -> None:
    import veldra.gui.app as app_module

    app = app_module.create_app()

    render = _wrapped(app, "page-content.children")
    assert render("/run") is not None

    set_poll = _wrapped(app, "run-jobs-interval.interval")
    assert set_poll("/run") >= 200

    enqueue_cb = _wrapped(app, "..run-result-json.children...run-result-log.children..")
    monkeypatch.setattr(
        app_module,
        "enqueue_run_job_result",
        lambda **_: ('{"job":"1"}', "[QUEUED] ok"),
    )
    payload, log = enqueue_cb(1, "fit", "", "", "", "", "", "python")
    assert '"job"' in payload
    assert "[QUEUED]" in log

    monkeypatch.setattr(
        app_module,
        "enqueue_run_job_result",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    payload_err, log_err = enqueue_cb(1, "fit", "", "", "", "", "", "python")
    assert payload_err == "{}"
    assert "[ERROR]" in log_err

    refresh_jobs_cb = _wrapped(
        app,
        "..run-job-select.options...run-job-select.value...run-jobs-table.children..",
    )
    monkeypatch.setattr(
        app_module,
        "build_job_options",
        lambda limit=100: ([{"label": "a", "value": "j1"}], "j1", "[]"),
    )
    opts, selected, table = refresh_jobs_cb(1, 0, "j1")
    assert selected == "j1"
    assert len(opts) == 1
    assert table == "[]"
    _opts2, selected2, _table2 = refresh_jobs_cb(1, 0, "other")
    assert selected2 == "j1"

    detail_cb = _wrapped(app, "run-job-detail.children")
    monkeypatch.setattr(app_module, "format_job_detail", lambda _job_id: "detail")
    assert detail_cb("j1", 1) == "detail"

    cancel_cb = _wrapped_by_prefix(app, "run-result-log.children@")
    assert cancel_cb(1, None).startswith("[ERROR]")
    monkeypatch.setattr(
        app_module,
        "cancel_run_job",
        lambda _job_id: types.SimpleNamespace(message="canceled"),
    )
    assert cancel_cb(1, "j1") == "[INFO] canceled"
    monkeypatch.setattr(
        app_module,
        "cancel_run_job",
        lambda _job_id: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert cancel_cb(1, "j1").startswith("[ERROR]")

    refresh_artifacts_cb = _wrapped(app, "..artifact-select.options...artifact-select.value..")
    monkeypatch.setattr(app_module, "build_artifact_options", lambda root: ([{"value": "a"}], "a"))
    assert refresh_artifacts_cb(1, "artifacts")[1] == "a"

    show_metrics_cb = _wrapped(app, "artifact-metrics.children")
    monkeypatch.setattr(app_module, "format_artifact_metrics", lambda _path: "metrics")
    assert show_metrics_cb("a") == "metrics"

    eval_artifact_cb = _wrapped(app, "artifact-eval-result.children")
    monkeypatch.setattr(app_module, "evaluate_selected_artifact", lambda _a, _d: '{"ok":true}')
    assert json.loads(eval_artifact_cb(1, "a", "d"))["ok"] is True


def test_config_callback_wrappers(monkeypatch) -> None:
    import veldra.gui.app as app_module

    app = app_module.create_app()

    config_cb = _wrapped(app, "..config-yaml.value...config-validation-result.children..")
    monkeypatch.setattr(
        app_module,
        "callback_context",
        types.SimpleNamespace(triggered=[{"prop_id": "config-load-btn.n_clicks"}]),
    )
    monkeypatch.setattr(app_module, "handle_config_action", lambda *_: ("x", "ok"))
    assert config_cb(1, 0, 0, "a", "b") == ("x", "ok")

    migrate_cb = _wrapped(
        app,
        "..config-migrate-normalized-yaml.value...config-migrate-diff.children...config-migrate-result.children..",
    )
    monkeypatch.setattr(
        app_module,
        "callback_context",
        types.SimpleNamespace(triggered=[{"prop_id": "config-migrate-preview-btn.n_clicks"}]),
    )
    monkeypatch.setattr(
        app_module,
        "handle_config_migrate_preview",
        lambda *_: ("norm", "diff", "res"),
    )
    assert migrate_cb(1, 0, "yaml", "in", "out", 1, "", "") == ("norm", "diff", "res")

    monkeypatch.setattr(
        app_module,
        "callback_context",
        types.SimpleNamespace(triggered=[{"prop_id": "config-migrate-apply-btn.n_clicks"}]),
    )
    monkeypatch.setattr(app_module, "handle_config_migrate_apply", lambda **_: "applied")
    assert migrate_cb(0, 1, "yaml", "in", "out", 1, "n", "d") == ("n", "d", "applied")

    monkeypatch.setattr(
        app_module,
        "handle_config_migrate_apply",
        lambda **_: (_ for _ in ()).throw(RuntimeError("migrate failed")),
    )
    out = migrate_cb(0, 1, "yaml", "in", "out", 1, "n2", "d2")
    assert out[0] == "n2"
    assert out[1] == "d2"
    assert out[2].startswith("RuntimeError:")
