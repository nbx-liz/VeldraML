from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from plotly.graph_objs import Figure

from veldra.gui import app as app_module


def test_lazy_runtime_getters(monkeypatch) -> None:
    original_artifact = app_module.Artifact
    original_cls = app_module._ARTIFACT_CLS
    original_eval = app_module.evaluate
    original_loader = app_module.load_tabular_data
    try:
        sentinel = object()
        app_module.Artifact = sentinel
        assert app_module._get_artifact_cls() is sentinel

        app_module.Artifact = app_module._ArtifactProxy
        app_module._ARTIFACT_CLS = None
        cls1 = app_module._get_artifact_cls()
        cls2 = app_module._get_artifact_cls()
        assert cls1 is cls2

        app_module.evaluate = None
        ev1 = app_module._get_evaluate()
        ev2 = app_module._get_evaluate()
        assert ev1 is ev2

        app_module.load_tabular_data = None
        ld1 = app_module._get_load_tabular_data()
        ld2 = app_module._get_load_tabular_data()
        assert ld1 is ld2
    finally:
        app_module.Artifact = original_artifact
        app_module._ARTIFACT_CLS = original_cls
        app_module.evaluate = original_eval
        app_module.load_tabular_data = original_loader


def test_ensure_default_run_config_and_render_routes(tmp_path: Path) -> None:
    cfg = tmp_path / "configs" / "gui.yaml"
    out = app_module._ensure_default_run_config(str(cfg))
    assert out == str(cfg)
    assert cfg.exists()

    # Existing path should remain untouched.
    cfg.write_text("task:\n  type: regression\n", encoding="utf-8")
    app_module._ensure_default_run_config(str(cfg))
    assert "regression" in cfg.read_text(encoding="utf-8")

    assert app_module.render_page("/data") is not None


def test_system_temp_dir_and_cleanup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module.tempfile, "gettempdir", lambda: str(tmp_path))
    tmp_dir = app_module._get_gui_system_temp_dir()
    assert tmp_dir.name == "veldra_system_temporary_uploads"
    assert tmp_dir.exists()

    stale_file = tmp_dir / "stale.csv"
    fresh_file = tmp_dir / "fresh.csv"
    stale_file.write_text("x", encoding="utf-8")
    fresh_file.write_text("y", encoding="utf-8")
    now = app_module.time.time()
    os.utime(stale_file, (now - 1000, now - 1000))
    os.utime(fresh_file, (now - 10, now - 10))
    app_module._cleanup_gui_system_temp_files(max_age_seconds=100)
    assert not stale_file.exists()
    assert fresh_file.exists()


def test_stepper_bar_and_timestamp_helpers() -> None:
    bar = app_module._stepper_bar("/config")
    assert bar.className == "stepper-container"
    assert len(bar.children) == 11

    root = app_module._stepper_bar("/")
    assert "active" in str(root)

    assert app_module._format_jst_timestamp(None) == "n/a"
    assert app_module._format_jst_timestamp("invalid") == "invalid"
    assert app_module._format_jst_timestamp("2026-01-01T00:00:00") == "2026-01-01 09:00:00 JST"


def test_stepper_connector_colors_follow_progress() -> None:
    run_bar = app_module._stepper_bar("/run")
    connector_lines = [
        child for child in run_bar.children if getattr(child, "style", {}).get("flexGrow") == "1"
    ]
    assert len(connector_lines) == 5
    assert connector_lines[0].style["backgroundColor"] == "var(--success)"
    assert connector_lines[1].style["backgroundColor"] == "var(--success)"
    assert connector_lines[2].style["backgroundColor"] == "var(--success)"
    assert connector_lines[3].style["backgroundColor"] == "var(--success)"
    assert connector_lines[4].style["backgroundColor"] == "rgba(148, 163, 184, 0.1)"


def test_cb_inspect_data_branches(monkeypatch) -> None:
    # Invalid base64 payload.
    bad = app_module._cb_inspect_data(1, "data:text/csv;base64", "a.csv", None, {})
    assert "Invalid file format" in bad[1]

    # Unsupported extension.
    import base64

    encoded = base64.b64encode(b"a,b\n1,2\n").decode("ascii")
    unsup = app_module._cb_inspect_data(1, f"data:text/plain;base64,{encoded}", "a.txt", None, {})
    assert "Unsupported file type" in unsup[1]

    # Filename selected but no contents.
    selected = app_module._cb_inspect_data(1, None, "picked.csv", None, {"x": 1})
    assert selected[1] == ""
    assert selected[3].startswith("Selected:")

    # Nothing selected.
    empty = app_module._cb_inspect_data(1, None, None, None, {})
    assert "Please select" in empty[1]

    # inspect_data failure.
    monkeypatch.setattr(app_module, "inspect_data", lambda _p: {"success": False, "error": "boom"})
    fail = app_module._cb_inspect_data(1, f"data:text/csv;base64,{encoded}", "a.csv", None, {})
    assert "Error: boom" in fail[1]


def test_small_state_callbacks() -> None:
    assert app_module._cb_save_target_col("y", None)["target_col"] == "y"
    assert app_module._cb_update_selected_file_label([])[0].startswith("No file selected")
    assert app_module._cb_update_selected_file_label(["a.csv"])[0] == "Selected: a.csv"
    assert app_module._cb_cache_config_yaml("x: 1", None)["config_yaml"] == "x: 1"


def test_config_related_callbacks_branches(monkeypatch, tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("config_version: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        app_module,
        "migrate_config_from_yaml",
        lambda content, target_version=1: (f"v{target_version}", "diff"),
    )
    ok = app_module._cb_handle_migration_preview(1, str(p), 2)
    assert ok == ("v2", "diff", None)

    miss = app_module._cb_handle_migration_preview(1, str(tmp_path / "missing.yaml"), 1)
    assert "File not found" in miss[2]

    monkeypatch.setattr(
        app_module,
        "migrate_config_from_yaml",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    err = app_module._cb_handle_migration_preview(1, str(p), 1)
    assert "Error:" in err[2]

    monkeypatch.setattr(app_module, "migrate_config_file_via_gui", lambda *_a, **_k: "applied")
    out = app_module._cb_handle_migration_apply(1, str(p), 1)
    assert "applied" in str(out)

    monkeypatch.setattr(
        app_module,
        "migrate_config_file_via_gui",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    out_err = app_module._cb_handle_migration_apply(1, str(p), 1)
    assert "Error:" in str(out_err)

    assert app_module._cb_update_split_options("group")[0]["display"] == "block"
    assert app_module._cb_update_split_options("timeseries")[1]["display"] == "block"
    warn_empty = app_module._cb_timeseries_time_warning("timeseries", None)
    assert "requires selecting Time Column" in warn_empty[0]
    assert warn_empty[1]["display"] == "block"
    warn_set = app_module._cb_timeseries_time_warning("timeseries", "ts")
    assert warn_set == ("", {"display": "none"})
    warn_not_ts = app_module._cb_timeseries_time_warning("kfold", "")
    assert warn_not_ts == ("", {"display": "none"})
    assert app_module._cb_update_tune_visibility(True)["display"] == "block"
    assert app_module._cb_update_tune_visibility(False)["display"] == "none"

    assert app_module._cb_detect_run_action("") == (
        "fit",
        "Ready: TRAIN",
        "badge bg-primary fs-6 p-2 mb-3",
    )
    assert app_module._cb_detect_run_action("- not: dict") == (
        "fit",
        "Ready: TRAIN",
        "badge bg-primary fs-6 p-2 mb-3",
    )
    assert app_module._cb_detect_run_action("tuning:\n  enabled: true\n")[0] == "tune"
    assert "CAUSAL" in app_module._cb_detect_run_action("task:\n  causal_method: dr\n")[1]

    assert app_module._cb_update_tune_objectives("regression")
    binary_opts = app_module._cb_update_tune_objectives("binary")
    assert any(opt["value"] == "brier" for opt in binary_opts)
    assert any(opt["value"] == "precision_at_k" for opt in binary_opts)
    assert app_module._cb_update_tune_objectives("unknown") == []
    assert app_module._cb_update_top_k_visibility("binary") == {"display": "block"}
    assert app_module._cb_update_top_k_visibility("regression") == {"display": "none"}


def test_populate_builder_options_modern_and_legacy(monkeypatch) -> None:
    # No state: modern return when callback context is available.
    monkeypatch.setattr(app_module, "callback_context", SimpleNamespace(triggered_id="url"))
    out = app_module._cb_populate_builder_options("/config", None)
    assert out == ("", "", [], [], [], [], [], [])

    # No state: legacy return when callback context is unavailable.
    class _NoCtx:
        @property
        def triggered_id(self):
            raise RuntimeError("no ctx")

    monkeypatch.setattr(app_module, "callback_context", _NoCtx())
    out_legacy = app_module._cb_populate_builder_options("/config", None)
    assert out_legacy == ("", "", [], [], [], [], [])

    monkeypatch.setattr(
        app_module,
        "inspect_data",
        lambda _p: {"success": True, "stats": {"columns": ["x", "y", "target"]}},
    )
    monkeypatch.setattr(app_module, "callback_context", SimpleNamespace(triggered_id="url"))
    state = {"data_path": "data.csv", "target_col": "target"}
    modern = app_module._cb_populate_builder_options("/config", state)
    assert modern[0] == "data.csv"
    assert len(modern) == 8

    # inspect_data exception fallback.
    monkeypatch.setattr(
        app_module,
        "inspect_data",
        lambda _p: (_ for _ in ()).throw(RuntimeError("inspect fail")),
    )
    legacy_ctx = app_module._cb_populate_builder_options("/config", state)
    assert legacy_ctx[3] == []


def test_build_config_yaml_branches() -> None:
    text = app_module._cb_build_config_yaml(
        "regression",
        "data.csv",
        "target",
        ["id"],
        ["cat"],
        ["drop"],
        "timeseries",
        5,
        42,
        "group_col",
        "ts",
        "expanding",
        0.2,
        2,
        1,
        0.1,
        31,
        100,
        -1,
        20,
        10,
        0.9,
        0.8,
        0.0,
        0.1,
        True,
        "fast",
        20,
        "rmse",
        "artifacts",
        True,
        "dr",
        0.01,
        0.1,
        8,
        64,
        3,
        10,
        0.5,
        0.9,
    )
    assert "causal_method" in text
    assert "timeseries_mode" in text
    assert "feature_fraction" in text
    assert "num_boost_round" in text
    assert "n_estimators" not in text


def test_run_launch_state_and_polling_extra(monkeypatch) -> None:
    # Hit evaluate/simulate data-missing branches.
    eval_missing = app_module._cb_update_run_launch_state("evaluate", "", "cfg", "", "", "")
    assert "Data Source" in eval_missing[1]
    sim_missing = app_module._cb_update_run_launch_state("simulate", "", "", "", "", "")
    assert "Data Source" in sim_missing[1]

    monkeypatch.setenv("VELDRA_GUI_POLL_MS", "50")
    assert app_module._cb_set_run_polling("/run") == 200


def test_show_job_detail_payload_and_status(monkeypatch) -> None:
    from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation

    succeeded = GuiJobRecord(
        job_id="j-ok",
        status="succeeded",
        action="fit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="fit"),
        result=GuiRunResult(True, "ok", {"x": 1}),
    )
    monkeypatch.setattr(app_module, "get_run_job", lambda _jid: succeeded)

    detail = app_module._cb_show_selected_job_detail(None, [{"job_id": "j-ok"}], "j-ok")
    assert "SUCCEEDED" in str(detail[0])
    assert "Result Payload" in str(detail[0])

    no_job = app_module._cb_show_selected_job_detail(None, [{"job_id": "j-ok"}], None)
    assert no_job[0].startswith("Select a job")


def test_result_view_additional_metric_selection_and_booster_exception(monkeypatch) -> None:
    class _ArtifactA:
        metrics = [1, 2]
        metadata = {}
        feature_schema = {}
        manifest = SimpleNamespace(run_id="r", task_type="regression", created_at_utc="2026-01-01")
        run_config = {"task": {"type": "regression"}}

    class _ArtifactB:
        metrics = {"mean": 1}
        metadata = {}
        feature_schema = {}
        manifest = SimpleNamespace(run_id="r2", task_type="regression", created_at_utc="2026-01-01")
        run_config = {"task": {"type": "regression"}}

        def _get_booster(self):
            raise RuntimeError("booster fail")

    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(load=lambda p: _ArtifactA() if "a" in p else _ArtifactB()),
    )
    monkeypatch.setattr(
        app_module,
        "plot_metrics_bar",
        lambda m, title=None: {"m": m, "title": title},
    )
    monkeypatch.setattr(app_module, "plot_feature_importance", lambda fi: {"fi": fi})
    monkeypatch.setattr(app_module, "kpi_card", lambda k, v: f"{k}:{v}")

    a = app_module._cb_update_result_view("a", None)
    assert a[1]["m"] == {}

    b = app_module._cb_update_result_view("b", None)
    assert isinstance(b[2], Figure)


def test_evaluate_action_asdict_fallback_and_data_pass(monkeypatch) -> None:
    class _ResultObj:
        __dataclass_fields__ = {"data": object(), "metrics": object()}

        def __init__(self) -> None:
            self.data = {"rows": 1}
            self.metrics = {"rmse": 1.0}

    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=lambda _p: object()))
    monkeypatch.setattr(app_module, "_get_load_tabular_data", lambda: (lambda _p: object()))
    monkeypatch.setattr(app_module, "_get_evaluate", lambda: (lambda _a, _f: _ResultObj()))
    monkeypatch.setattr(app_module, "asdict", lambda _x: (_ for _ in ()).throw(RuntimeError("x")))

    out = app_module._cb_evaluate_artifact_action(1, "artifact", "data")
    assert '"data"' in out
    assert '"metrics"' in out
