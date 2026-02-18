from __future__ import annotations

from types import SimpleNamespace

import dash
import pandas as pd

from veldra.gui import app as app_module


def test_validation_guardrails_branches(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "render_guardrails", lambda payload: payload)

    monkeypatch.setattr(app_module, "_load_data_from_state", lambda _state: None)
    no_data = app_module._cb_validation_guardrails("kfold", 5, None, None, {})
    assert no_data[0]["message"] == "Load data first."

    frame = pd.DataFrame({"target": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    monkeypatch.setattr(app_module, "_load_data_from_state", lambda _state: frame)
    state = {"task_type": "binary", "causal_config": {"enabled": True}, "exclude_cols": []}
    findings = app_module._cb_validation_guardrails("kfold", 5, None, None, state)
    text = "\n".join(str(item.get("message")) for item in findings)
    assert "Classification tasks" in text

    stratified = app_module._cb_validation_guardrails("stratified", 5, None, None, state)
    text2 = "\n".join(str(item.get("message")) for item in stratified)
    assert "Causal workflows" in text2


def test_validation_recommendation_timeseries_badge() -> None:
    out = app_module._cb_validation_recommendation(
        {
            "task_type": "regression",
            "split_config": {"type": "timeseries"},
            "causal_config": {"enabled": False},
        }
    )
    assert "TimeSeries split selected" in str(out)


def test_apply_train_preset_balanced_and_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "train-preset-balanced-btn.n_clicks"}]),
    )
    assert app_module._cb_apply_train_preset(0, 1, 0.1, 20, 10) == (0.05, 63, 20)

    class _BrokenCtx:
        @property
        def triggered(self):
            raise RuntimeError("no ctx")

    monkeypatch.setattr(app_module, "callback_context", _BrokenCtx())
    assert app_module._cb_apply_train_preset(0, 0, None, None, None) == (0.05, 31, 20)


def test_train_objective_help_empty_and_default() -> None:
    assert app_module._cb_train_objective_help(None) == ""
    default_help = app_module._cb_train_objective_help("unknown_objective")
    assert "OBJECTIVE" in str(default_help).upper()


def test_train_yaml_actions_all_paths(monkeypatch) -> None:
    base_state = {"task_type": "regression", "data_path": "d.csv", "target_col": "y"}

    monkeypatch.setattr(app_module, "load_config_yaml", lambda _p: "task:\n  type: regression\n")
    monkeypatch.setattr(
        app_module,
        "_state_from_config_payload",
        lambda payload, current: {**current, "payload": payload},
    )
    monkeypatch.setattr(app_module, "save_config_yaml", lambda _p, _y: "saved.yml")
    monkeypatch.setattr(app_module, "validate_config", lambda _y: object())

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "train-config-load-btn.n_clicks"}]),
    )
    out_yaml, msg, state = app_module._cb_train_yaml_actions(1, 0, 0, 0, "", "cfg.yml", base_state)
    assert "Loaded" in msg
    assert state["config_yaml"] == out_yaml

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "train-config-save-btn.n_clicks"}]),
    )
    _, msg_save, _ = app_module._cb_train_yaml_actions(0, 1, 0, 0, "a: 1\n", "cfg.yml", base_state)
    assert "Saved" in msg_save

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "train-config-validate-btn.n_clicks"}]),
    )
    _, msg_validate, _ = app_module._cb_train_yaml_actions(
        0, 0, 1, 0, "a: 1\n", "cfg.yml", base_state
    )
    assert "passed" in msg_validate

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "train-config-yaml-import-btn.n_clicks"}]),
    )
    _, msg_import, state_import = app_module._cb_train_yaml_actions(
        0,
        0,
        0,
        1,
        "task:\n  type: regression\n",
        "cfg.yml",
        base_state,
    )
    assert "Imported" in msg_import
    assert "payload" in state_import

    _, msg_error, _ = app_module._cb_train_yaml_actions(
        0, 0, 0, 1, "- a\n- b\n", "cfg.yml", base_state
    )
    assert "Error" in msg_error


def test_sync_run_inputs_and_override_from_state() -> None:
    non_run = app_module._cb_sync_run_inputs_from_state("/train", {"x": 1})
    assert non_run == (dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    on_run = app_module._cb_sync_run_inputs_from_state(
        "/run",
        {"data_path": "d.csv", "config_yaml": "a: 1\n", "artifact_dir": "output"},
    )
    assert on_run == ("d.csv", "a: 1\n", "configs/gui_run.yaml", "output")

    non_run_override = app_module._cb_sync_run_override_from_state("/results", {})
    assert non_run_override == (dash.no_update, dash.no_update)

    on_run_override = app_module._cb_sync_run_override_from_state(
        "/run",
        {"run_action_override": {"mode": "manual", "action": "simulate"}},
    )
    assert on_run_override == ("manual", "simulate")
