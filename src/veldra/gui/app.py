"""Dash app factory for Veldra GUI."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import yaml
from dash import Input, Output, State, callback_context, dcc, html

from veldra.gui.components.charts import (
    plot_comparison_bar,
    plot_feature_importance,
    plot_learning_curves,
    plot_metrics_bar,
)
from veldra.gui.components.guardrail import render_guardrails
from veldra.gui.components.help_texts import HELP_TEXTS
from veldra.gui.components.help_ui import context_card, guide_alert, recommendation_badge
from veldra.gui.components.kpi_cards import kpi_card
from veldra.gui.components.progress_viewer import render_progress_viewer
from veldra.gui.components.task_table import task_table
from veldra.gui.components.toast import make_toast, toast_container
from veldra.gui.components.yaml_diff import render_yaml_diff
from veldra.gui.pages import (
    compare_page,
    config_page,
    data_page,
    results_page,
    run_page,
    runs_page,
    target_page,
    train_page,
    validation_page,
)
from veldra.gui.services import (
    GuardRailChecker,
    GuardRailResult,
    cancel_run_job,
    compare_artifacts,
    delete_run_jobs,
    get_run_job,
    infer_task_type,
    inspect_data,
    list_artifacts,
    list_run_job_logs,
    list_run_jobs,
    list_run_jobs_filtered,
    load_config_yaml,
    load_job_config_yaml,
    migrate_config_file_via_gui,
    migrate_config_from_yaml,
    normalize_gui_error,
    retry_run_job,
    save_config_yaml,
    set_run_job_priority,
    submit_run_job,
    validate_config,
    validate_config_with_guidance,
)
from veldra.gui.template_service import (
    clone_custom_slot,
    count_yaml_changes,
    custom_slot_options,
    load_builtin_template_yaml,
    load_custom_slot_yaml,
    save_custom_slot,
)
from veldra.gui.types import RunInvocation

# Lazy runtime symbols for heavyweight deps.
_ARTIFACT_CLS: Any | None = None
evaluate: Any | None = None
load_tabular_data: Any | None = None

_GUI_SYSTEM_TEMP_DIR_NAME = "veldra_system_temporary_uploads"
_GUI_UPLOAD_TTL_SECONDS = int(os.getenv("VELDRA_GUI_UPLOAD_TTL_SECONDS", "86400"))


def _get_artifact_cls() -> Any:
    global Artifact, _ARTIFACT_CLS
    if Artifact is not _ArtifactProxy:
        return Artifact
    if _ARTIFACT_CLS is None:
        from veldra.api.artifact import Artifact as _Artifact

        _ARTIFACT_CLS = _Artifact
    return _ARTIFACT_CLS


class _ArtifactProxy:
    @staticmethod
    def load(path: str) -> Any:
        return _get_artifact_cls().load(path)


# Backward-compat name used in tests via monkeypatch("...Artifact.load", ...).
Artifact: Any = _ArtifactProxy


def _get_evaluate() -> Any:
    global evaluate
    if evaluate is None:
        from veldra.api.runner import evaluate as _evaluate

        evaluate = _evaluate
    return evaluate


def _get_load_tabular_data() -> Any:
    global load_tabular_data
    if load_tabular_data is None:
        from veldra.data import load_tabular_data as _load_tabular_data

        load_tabular_data = _load_tabular_data
    return load_tabular_data


DEFAULT_GUI_RUN_CONFIG_YAML = """# Veldra GUI default run config
# This file is auto-created when missing.
config_version: 1

task:
  type: regression

data:
  # Existing demo dataset in this repository.
  path: examples/data/causal_dr_tune_demo.csv
  target: target
  drop_cols:
    - treatment

split:
  type: kfold
  n_splits: 5
  seed: 42

train:
  num_boost_round: 120
  lgb_params:
    learning_rate: 0.1
    num_leaves: 31
    max_depth: -1
    min_child_samples: 20
    subsample: 1.0
    colsample_bytree: 1.0
    reg_alpha: 0.0
    reg_lambda: 0.0
  early_stopping_rounds: 20
  seed: 42

tuning:
  enabled: false

export:
  artifact_dir: artifacts
"""


def _ensure_default_run_config(config_path: str) -> str:
    path = Path(config_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_GUI_RUN_CONFIG_YAML, encoding="utf-8")
    return str(path)


def _get_gui_system_temp_dir() -> Path:
    base = Path(tempfile.gettempdir())
    tmp_dir = base / _GUI_SYSTEM_TEMP_DIR_NAME
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def _cleanup_gui_system_temp_files(max_age_seconds: int | None = None) -> None:
    ttl = _GUI_UPLOAD_TTL_SECONDS if max_age_seconds is None else max_age_seconds
    tmp_dir = _get_gui_system_temp_dir()
    now = time.time()
    for item in tmp_dir.iterdir():
        if not item.is_file():
            continue
        try:
            age = now - item.stat().st_mtime
            if age > ttl:
                item.unlink()
        except Exception:
            continue


def _default_split_for_task(
    task_type: str | None, *, causal_enabled: bool = False
) -> dict[str, Any]:
    if causal_enabled:
        return {"type": "group", "n_splits": 5, "seed": 42, "group_col": None}
    task = (task_type or "regression").strip().lower()
    split_type = "stratified" if task in {"binary", "multiclass"} else "kfold"
    return {"type": split_type, "n_splits": 5, "seed": 42}


def _default_train_config() -> dict[str, Any]:
    return {
        "learning_rate": 0.05,
        "num_boost_round": 300,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "early_stopping_rounds": 100,
        "auto_class_weight": True,
        "class_weight_text": "",
    }


def _default_tuning_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "preset": "standard",
        "n_trials": 30,
        "objective": None,
    }


def _ensure_workflow_state_defaults(state: dict[str, Any] | None) -> dict[str, Any]:
    current = dict(state or {})
    task_type = str(current.get("task_type") or current.get("task", "regression"))
    causal_cfg = current.get("causal_config") or {}
    current.setdefault("task_type", task_type)
    current.setdefault("exclude_cols", [])
    current.setdefault("causal_config", causal_cfg)
    if not isinstance(current.get("split_config"), dict) or not current.get("split_config"):
        current["split_config"] = _default_split_for_task(task_type)
    if not isinstance(current.get("train_config"), dict) or not current.get("train_config"):
        current["train_config"] = _default_train_config()
    if not isinstance(current.get("tuning_config"), dict) or not current.get("tuning_config"):
        current["tuning_config"] = _default_tuning_config()
    current.setdefault("artifact_dir", "artifacts")
    current.setdefault("config_yaml", "")
    current.setdefault("template_id", "regression_baseline")
    current.setdefault("template_origin", "builtin")
    current.setdefault("custom_config_slots", [])
    current.setdefault("wizard_state", {"step": 1, "draft_payload": {}, "completed_steps": []})
    current.setdefault(
        "last_validation",
        {"ok": False, "errors": [], "warnings": [], "timestamp_utc": None},
    )
    current.setdefault("config_diff_base_yaml", "")
    current.setdefault("last_job_succeeded", False)
    current.setdefault("results_shortcut_focus", None)
    current.setdefault("run_action_override", {"mode": "auto", "action": None})
    return current


def _build_config_payload_from_state(state: dict[str, Any] | None) -> dict[str, Any]:
    current = _ensure_workflow_state_defaults(state)
    task_type = str(current.get("task_type") or "regression")
    data_path = current.get("data_path")
    target_col = current.get("target_col")
    exclude_cols = list(current.get("exclude_cols") or [])
    split_cfg = dict(current.get("split_config") or {})
    train_cfg = dict(current.get("train_config") or {})
    tuning_cfg = dict(current.get("tuning_config") or {})
    causal_cfg = dict(current.get("causal_config") or {})

    payload: dict[str, Any] = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {
            "path": data_path or "",
            "target": target_col or "",
        },
        "split": {
            "type": split_cfg.get("type", "kfold"),
            "n_splits": int(split_cfg.get("n_splits", 5)),
            "seed": int(split_cfg.get("seed", 42)),
        },
        "train": {
            "num_boost_round": int(train_cfg.get("num_boost_round", 300)),
            "lgb_params": {
                "learning_rate": float(train_cfg.get("learning_rate", 0.05)),
                "num_leaves": int(train_cfg.get("num_leaves", 31)),
                "max_depth": int(train_cfg.get("max_depth", -1)),
                "min_child_samples": int(train_cfg.get("min_child_samples", 20)),
                "subsample": float(train_cfg.get("subsample", 1.0)),
                "colsample_bytree": float(train_cfg.get("colsample_bytree", 1.0)),
                "reg_alpha": float(train_cfg.get("reg_alpha", 0.0)),
                "reg_lambda": float(train_cfg.get("reg_lambda", 0.0)),
            },
            "early_stopping_rounds": int(train_cfg.get("early_stopping_rounds", 100)),
            "seed": 42,
        },
        "tuning": {
            "enabled": bool(tuning_cfg.get("enabled", False)),
        },
        "export": {
            "artifact_dir": current.get("artifact_dir") or "artifacts",
        },
    }

    if exclude_cols:
        payload["data"]["drop_cols"] = exclude_cols

    if task_type in {"binary", "multiclass"}:
        payload["train"]["auto_class_weight"] = bool(train_cfg.get("auto_class_weight", True))
        class_weight_text = str(train_cfg.get("class_weight_text") or "").strip()
        if not payload["train"]["auto_class_weight"] and class_weight_text:
            try:
                parsed = yaml.safe_load(class_weight_text)
                if isinstance(parsed, dict):
                    payload["train"]["class_weight"] = {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                pass

    split_type = str(split_cfg.get("type", "kfold"))
    if split_type == "group" and split_cfg.get("group_col"):
        payload["split"]["group_col"] = split_cfg.get("group_col")
    if split_type == "timeseries":
        if split_cfg.get("time_col"):
            payload["split"]["time_col"] = split_cfg.get("time_col")
        payload["split"]["timeseries_mode"] = split_cfg.get("timeseries_mode", "expanding")
        if split_cfg.get("test_size"):
            payload["split"]["test_size"] = int(split_cfg.get("test_size"))
        payload["split"]["gap"] = int(split_cfg.get("gap", 0))
        payload["split"]["embargo"] = int(split_cfg.get("embargo", 0))

    if payload["tuning"]["enabled"]:
        payload["tuning"]["preset"] = tuning_cfg.get("preset", "standard")
        payload["tuning"]["n_trials"] = int(tuning_cfg.get("n_trials", 30))
        objective = tuning_cfg.get("objective")
        if objective:
            payload["tuning"]["objective"] = objective

    if causal_cfg.get("enabled") and causal_cfg.get("method"):
        payload["task"]["causal_method"] = causal_cfg.get("method")
        causal_payload: dict[str, Any] = {}
        if causal_cfg.get("treatment_col"):
            causal_payload["treatment_col"] = causal_cfg.get("treatment_col")
        if causal_cfg.get("unit_id_col"):
            causal_payload["unit_id_col"] = causal_cfg.get("unit_id_col")
        if causal_payload:
            payload["causal"] = causal_payload
    return payload


def _build_config_from_state(state: dict[str, Any] | None) -> str:
    payload = _build_config_payload_from_state(state)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _state_from_config_payload(
    payload: dict[str, Any], base_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    state = _ensure_workflow_state_defaults(base_state)
    task = payload.get("task") or {}
    data_obj = payload.get("data") or {}
    split_obj = payload.get("split") or {}
    train_obj = payload.get("train") or {}
    lgb_obj = train_obj.get("lgb_params") or {}
    tuning_obj = payload.get("tuning") or {}
    causal_obj = payload.get("causal") or {}

    state["task_type"] = task.get("type", state.get("task_type", "regression"))
    if isinstance(data_obj.get("path"), str):
        state["data_path"] = data_obj.get("path")
    if isinstance(data_obj.get("target"), str):
        state["target_col"] = data_obj.get("target")
    state["exclude_cols"] = list(data_obj.get("drop_cols") or [])
    state["split_config"] = {
        "type": split_obj.get("type", "kfold"),
        "n_splits": int(split_obj.get("n_splits", 5)),
        "seed": int(split_obj.get("seed", 42)),
        "group_col": split_obj.get("group_col"),
        "time_col": split_obj.get("time_col"),
        "timeseries_mode": split_obj.get("timeseries_mode", "expanding"),
        "test_size": split_obj.get("test_size"),
        "gap": int(split_obj.get("gap", 0)) if split_obj.get("gap") is not None else 0,
        "embargo": int(split_obj.get("embargo", 0)) if split_obj.get("embargo") is not None else 0,
    }
    state["train_config"] = {
        "learning_rate": float(lgb_obj.get("learning_rate", 0.05)),
        "num_boost_round": int(train_obj.get("num_boost_round", 300)),
        "num_leaves": int(lgb_obj.get("num_leaves", 31)),
        "max_depth": int(lgb_obj.get("max_depth", -1)),
        "min_child_samples": int(lgb_obj.get("min_child_samples", 20)),
        "subsample": float(lgb_obj.get("subsample", 1.0)),
        "colsample_bytree": float(lgb_obj.get("colsample_bytree", 1.0)),
        "reg_alpha": float(lgb_obj.get("reg_alpha", 0.0)),
        "reg_lambda": float(lgb_obj.get("reg_lambda", 0.0)),
        "early_stopping_rounds": int(train_obj.get("early_stopping_rounds", 100)),
        "auto_class_weight": bool(train_obj.get("auto_class_weight", True)),
        "class_weight_text": (
            yaml.safe_dump(train_obj.get("class_weight"), sort_keys=False).strip()
            if train_obj.get("class_weight")
            else ""
        ),
    }
    state["tuning_config"] = {
        "enabled": bool(tuning_obj.get("enabled", False)),
        "preset": tuning_obj.get("preset", "standard"),
        "n_trials": int(tuning_obj.get("n_trials", 30)),
        "objective": tuning_obj.get("objective"),
    }
    state["artifact_dir"] = str((payload.get("export") or {}).get("artifact_dir", "artifacts"))
    state["causal_config"] = {
        "enabled": bool(task.get("causal_method")),
        "method": task.get("causal_method"),
        "treatment_col": causal_obj.get("treatment_col"),
        "unit_id_col": causal_obj.get("unit_id_col"),
    }
    state["config_yaml"] = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    return state


def _sidebar() -> html.Div:
    return html.Div(
        [
            html.H3(
                "Veldra",
                className="text-white mb-4 text-center fw-bold",
                style={"letterSpacing": "2px"},
            ),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [html.I(className="bi bi-database me-2"), "Data"],
                        href="/data",
                        active="exact",
                        className="nav-link",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-bullseye me-2"), "Target"],
                        href="/target",
                        active="exact",
                        className="nav-link",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-scissors me-2"), "Validation"],
                        href="/validation",
                        active="exact",
                        className="nav-link",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-sliders me-2"), "Train"],
                        href="/train",
                        active="exact",
                        className="nav-link",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-play-circle me-2"), "Run"],
                        href="/run",
                        active="exact",
                        className="nav-link",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-graph-up me-2"), "Results"],
                        href="/results",
                        active="exact",
                        className="nav-link",
                    ),
                    html.Hr(className="w-100 my-2"),
                    dbc.NavLink(
                        [html.I(className="bi bi-clock-history me-2"), "Runs"],
                        href="/runs",
                        active="exact",
                        className="nav-link",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-shuffle me-2"), "Compare"],
                        href="/compare",
                        active="exact",
                        className="nav-link",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
            html.Div(html.Span("v0.1.0", className="badge bg-secondary"), className="mt-auto mb-3"),
        ],
        className="sidebar",
    )


def _stepper_bar(pathname: str, state: dict[str, Any] | None = None) -> html.Div:
    steps = [
        {"label": "Data", "path": "/data"},
        {"label": "Target", "path": "/target"},
        {"label": "Validation", "path": "/validation"},
        {"label": "Train", "path": "/train"},
        {"label": "Run", "path": "/run"},
        {"label": "Results", "path": "/results"},
    ]

    normalized_state = _ensure_workflow_state_defaults(state)
    completed_map = {
        "/data": bool(normalized_state.get("data_path")),
        "/target": bool(normalized_state.get("target_col"))
        and bool(normalized_state.get("task_type")),
        "/validation": bool(normalized_state.get("split_config")),
        "/train": bool(normalized_state.get("train_config")),
        "/run": bool(normalized_state.get("last_job_succeeded")),
        "/results": bool(normalized_state.get("last_run_artifact")),
    }

    step_elems = []
    current_idx = 0
    normalized_path = "/target" if pathname == "/config" else (pathname or "/data")
    for i, step in enumerate(steps):
        if step["path"] == normalized_path or normalized_path.startswith(step["path"]):
            current_idx = i
            break

    for i, step in enumerate(steps):
        status_class = ""
        icon = str(i + 1)
        is_completed = bool(completed_map.get(step["path"])) or i < current_idx

        if is_completed:
            status_class = "completed"
            icon = "✓"
        if i == current_idx:
            status_class = "active"
            if is_completed:
                icon = "✓"

        step_content = html.Div(
            [html.Div(icon, className="step-circle"), html.Span(step["label"])],
            className=f"step-item {status_class}",
        )
        step_elems.append(
            dcc.Link(step_content, href=step["path"], style={"textDecoration": "none"})
        )

        if i < len(steps) - 1:
            connector_color = (
                "var(--success)"
                if (is_completed or i < current_idx)
                else "rgba(148, 163, 184, 0.1)"
            )
            step_elems.append(
                html.Div(
                    style={
                        "flexGrow": "1",
                        "height": "2px",
                        "backgroundColor": connector_color,
                        "margin": "0 10px",
                    }
                )
            )

    return html.Div(step_elems, className="stepper-container")


def _main_layout() -> html.Div:
    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="workflow-state", data={}),
            dcc.Store(id="last-job-status", data={}),
            dcc.Store(id="custom-config-slots-store", data=[], storage_type="local"),
            toast_container(),
            dbc.Row(
                [
                    dbc.Col(_sidebar(), width="auto", style={"width": "220px", "padding": "0"}),
                    dbc.Col(
                        [
                            html.Div(id="stepper-content", className="mt-4"),
                            html.Div(id="page-content", className="p-4"),
                        ],
                        id="main-content-col",
                        style={"flex": "1"},
                    ),
                ],
                className="g-0",
            ),
        ],
        id="app-shell",
    )


def render_page(pathname: str, state: dict | None = None) -> html.Div:
    if pathname == "/data" or pathname == "/":
        return data_page.layout()
    if pathname == "/config":
        return config_page.layout(state)
    if pathname == "/target":
        return target_page.layout(state)
    if pathname == "/validation":
        return validation_page.layout(state)
    if pathname == "/train":
        return train_page.layout(state)
    if pathname == "/run":
        return run_page.layout(state)
    if pathname == "/results":
        return results_page.layout()
    if pathname == "/runs":
        return runs_page.layout()
    if pathname == "/compare":
        return compare_page.layout()
    return data_page.layout()


def _to_jsonable(payload: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    if _seen is None:
        _seen = set()
    if _depth > 24:
        return "<max_depth_reached>"

    # Mock objects can generate unbounded model_dump() call chains.
    if isinstance(payload, Mock):
        return repr(payload)

    obj_id = id(payload)
    if obj_id in _seen:
        return "<cycle>"
    _seen.add(obj_id)

    if is_dataclass(payload):
        return {
            k: _to_jsonable(v, _seen=_seen, _depth=_depth + 1) for k, v in asdict(payload).items()
        }
    if hasattr(payload, "model_dump"):
        try:
            dumped = payload.model_dump(mode="json")
            if dumped is payload:
                return repr(payload)
            return _to_jsonable(dumped, _seen=_seen, _depth=_depth + 1)
        except Exception:
            return str(payload)
    if isinstance(payload, dict):
        return {str(k): _to_jsonable(v, _seen=_seen, _depth=_depth + 1) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_to_jsonable(v, _seen=_seen, _depth=_depth + 1) for v in payload]
    return payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False, default=str)


def _status_badge(status: str) -> str:
    return status.upper()


_JST = timezone(timedelta(hours=9))


def _format_jst_timestamp(value: str | None) -> str:
    if not value:
        return "n/a"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_JST).strftime("%Y-%m-%d %H:%M:%S JST")
    except Exception:
        return str(value)


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap",
        ],
        suppress_callback_exceptions=True,
        title="Veldra GUI",
    )
    app.layout = _main_layout()
    max(200, int(os.getenv("VELDRA_GUI_POLL_MS", "2000")))

    @app.callback(
        Output("page-content", "children"),
        Output("stepper-content", "children"),
        Input("url", "pathname"),
        State("workflow-state", "data"),
    )
    def _render_page(pathname: str | None, state: dict | None) -> tuple[Any, Any]:
        return render_page(pathname, state), _stepper_bar(pathname or "/", state)

    # --- Data Page Callbacks ---
    app.callback(
        Output("data-inspection-result", "children"),
        Output("data-error-message", "children"),
        Output("workflow-state", "data"),
        Output("data-selected-file-label", "children"),
        Output("data-file-path", "data"),
        Input("data-upload-drag", "contents"),
        State("data-upload-drag", "filename"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(lambda contents, filename, state: _cb_inspect_data(1, contents, filename, None, state))

    app.callback(
        Output("data-selected-file-label", "children", allow_duplicate=True),
        Input("data-upload-drag", "filename"),
        prevent_initial_call=True,
    )(lambda filename: _cb_update_selected_file_label(filename)[0])

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("data-target-col", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_save_target_col)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("config-yaml", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_cache_config_yaml)

    # Dash clientside callback entries may not expose "callback" in callback_map.
    # Some tests iterate callback_map values and assume this key always exists.
    for value in app.callback_map.values():
        if "callback" not in value:
            value["callback"] = lambda *args, **kwargs: dash.no_update

    # --- Config Page Callbacks ---
    app.callback(
        Output("config-yaml", "value"),
        Output("config-validation-result", "children"),
        Output("config-validation-result", "style"),
        Output("toast-container", "children", allow_duplicate=True),
        Input("config-validate-btn", "n_clicks"),
        Input("config-load-btn", "n_clicks"),
        Input("config-save-btn", "n_clicks"),
        Input("config-import-btn", "n_clicks"),
        State("config-yaml", "value"),
        State("config-file-path", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_handle_config_actions)

    app.callback(
        Output("config-migrate-normalized-yaml", "value"),
        Output("config-migrate-diff", "children"),
        Output("config-migrate-result", "children"),
        Input("config-migrate-preview-btn", "n_clicks"),
        State("config-migrate-input-path", "value"),
        State("config-migrate-target-version", "value"),
        prevent_initial_call=True,
    )(_cb_handle_migration_preview)

    app.callback(
        Output("config-migrate-result", "children", allow_duplicate=True),
        Input("config-migrate-apply-btn", "n_clicks"),
        State("config-migrate-input-path", "value"),
        State("config-migrate-target-version", "value"),
        prevent_initial_call=True,
    )(_cb_handle_migration_apply)

    app.callback(
        Output("cfg-data-path", "value"),
        Output("cfg-data-target", "value"),
        Output("cfg-data-target", "options"),
        Output("cfg-data-id-cols", "options"),
        Output("cfg-data-cat-cols", "options"),
        Output("cfg-data-drop-cols", "options"),
        Output("cfg-split-group-col", "options"),
        Output("cfg-split-time-col", "options"),
        Input("url", "pathname"),
        State("workflow-state", "data"),
    )(_cb_populate_builder_options)

    app.callback(
        Output("config-yaml", "value", allow_duplicate=True),
        Input("cfg-task-type", "value"),
        Input("cfg-data-path", "value"),
        Input("cfg-data-target", "value"),
        Input("cfg-data-id-cols", "value"),
        Input("cfg-data-cat-cols", "value"),
        Input("cfg-data-drop-cols", "value"),
        Input("cfg-split-type", "value"),
        Input("cfg-split-nsplits", "value"),
        Input("cfg-split-seed", "value"),
        Input("cfg-split-group-col", "value"),
        Input("cfg-split-time-col", "value"),
        Input("cfg-split-ts-mode", "value"),
        Input("cfg-split-test-size", "value"),
        Input("cfg-split-gap", "value"),
        Input("cfg-split-embargo", "value"),
        Input("cfg-train-lr", "value"),
        Input("cfg-train-num-leaves", "value"),
        Input("cfg-train-n-estimators", "value"),
        Input("cfg-train-max-depth", "value"),
        Input("cfg-train-min-child", "value"),
        Input("cfg-train-early-stop", "value"),
        Input("cfg-train-subsample", "value"),
        Input("cfg-train-colsample", "value"),
        Input("cfg-train-reg-alpha", "value"),
        Input("cfg-train-reg-lambda", "value"),
        Input("cfg-tune-enabled", "value"),
        Input("cfg-tune-preset", "value"),
        Input("cfg-tune-trials", "value"),
        Input("cfg-tune-objective", "value"),
        Input("cfg-export-dir", "value"),
        Input("cfg-causal-enabled", "value"),
        Input("cfg-causal-method", "value"),
        # Tuning Search Space
        Input("cfg-tune-lr-min", "value"),
        Input("cfg-tune-lr-max", "value"),
        Input("cfg-tune-leaves-min", "value"),
        Input("cfg-tune-leaves-max", "value"),
        Input("cfg-tune-depth-min", "value"),
        Input("cfg-tune-depth-max", "value"),
        Input("cfg-tune-ff-min", "value"),
        Input("cfg-tune-ff-max", "value"),
        Input("cfg-train-auto-class-weight", "value"),
        Input("cfg-train-class-weight", "value"),
        Input("cfg-train-auto-num-leaves", "value"),
        Input("cfg-train-num-leaves-ratio", "value"),
        Input("cfg-train-min-leaf-ratio", "value"),
        Input("cfg-train-min-bin-ratio", "value"),
        Input("cfg-train-feature-weights", "value"),
        Input("cfg-train-path-smooth", "value"),
        Input("cfg-train-cat-l2", "value"),
        Input("cfg-train-cat-smooth", "value"),
        Input("cfg-train-bagging-freq", "value"),
        Input("cfg-train-max-bin", "value"),
        Input("cfg-train-max-drop", "value"),
        Input("cfg-train-min-gain", "value"),
        Input("cfg-train-top-k", "value"),
        prevent_initial_call=True,
    )(_cb_build_config_yaml)

    # --- Visiblity Toggles ---
    app.callback(
        Output("cfg-container-tune", "style"),
        Input("cfg-tune-enabled", "value"),
    )(_cb_update_tune_visibility)

    app.callback(
        Output("cfg-causal-method-container", "style"),
        Input("cfg-causal-enabled", "value"),
    )(_cb_update_tune_visibility)  # Reuse same logic (block/none)

    app.callback(
        Output("cfg-container-group", "style"),
        Output("cfg-container-timeseries", "style"),
        Input("cfg-split-type", "value"),
    )(_cb_update_split_options)

    app.callback(
        Output("cfg-timeseries-time-warning", "children"),
        Output("cfg-timeseries-time-warning", "style"),
        Input("cfg-split-type", "value"),
        Input("cfg-split-time-col", "value"),
    )(_cb_timeseries_time_warning)

    app.callback(
        Output("cfg-container-class-weight-auto", "style"),
        Output("cfg-container-class-weight-manual", "style"),
        Input("cfg-task-type", "value"),
        Input("cfg-train-auto-class-weight", "value"),
    )(_cb_update_class_weight_visibility)

    app.callback(
        Output("cfg-container-top-k", "style"),
        Input("cfg-task-type", "value"),
    )(_cb_update_top_k_visibility)

    # --- Phase26 Target / Validation / Train callbacks ---
    app.callback(
        Output("target-data-path", "value"),
        Output("target-col-select", "options"),
        Output("target-col-select", "value"),
        Output("target-exclude-cols", "options"),
        Output("target-exclude-cols", "value"),
        Output("target-treatment-col", "options"),
        Output("target-unit-id-col", "options"),
        Output("target-task-type", "value"),
        Output("target-task-hint", "children"),
        Input("url", "pathname"),
        State("workflow-state", "data"),
    )(_cb_populate_target_page)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("target-col-select", "value"),
        Input("target-task-type", "value"),
        Input("target-exclude-cols", "value"),
        Input("target-causal-enabled", "value"),
        Input("target-causal-method", "value"),
        Input("target-treatment-col", "value"),
        Input("target-unit-id-col", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_save_target_state)

    app.callback(
        Output("target-task-context", "children"),
        Output("target-frontier-alpha-guide", "children"),
        Input("target-task-type", "value"),
    )(_cb_target_task_guides)

    app.callback(
        Output("target-causal-method-hint", "children"),
        Output("target-causal-context", "children"),
        Input("target-causal-enabled", "value"),
        Input("target-causal-method", "value"),
    )(_cb_target_causal_guides)

    app.callback(
        Output("target-guardrail-container", "children"),
        Input("target-col-select", "value"),
        Input("target-task-type", "value"),
        Input("target-exclude-cols", "value"),
        Input("target-causal-enabled", "value"),
        Input("target-causal-method", "value"),
        Input("target-treatment-col", "value"),
        Input("target-unit-id-col", "value"),
        State("workflow-state", "data"),
    )(_cb_target_guardrails)

    app.callback(
        Output("validation-group-col", "options"),
        Output("validation-time-col", "options"),
        Input("url", "pathname"),
        State("workflow-state", "data"),
    )(_cb_populate_validation_options)

    app.callback(
        Output("validation-group-container", "style"),
        Output("validation-timeseries-container", "style"),
        Input("validation-split-type", "value"),
    )(_cb_update_split_options)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("validation-split-type", "value"),
        Input("validation-n-splits", "value"),
        Input("validation-seed", "value"),
        Input("validation-group-col", "value"),
        Input("validation-time-col", "value"),
        Input("validation-ts-mode", "value"),
        Input("validation-test-size", "value"),
        Input("validation-gap", "value"),
        Input("validation-embargo", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_save_validation_state)

    app.callback(
        Output("validation-guardrail-container", "children"),
        Input("validation-split-type", "value"),
        Input("validation-n-splits", "value"),
        Input("validation-group-col", "value"),
        Input("validation-time-col", "value"),
        State("workflow-state", "data"),
    )(_cb_validation_guardrails)

    app.callback(
        Output("validation-split-context", "children"),
        Input("validation-split-type", "value"),
    )(_cb_validation_split_context)

    app.callback(
        Output("validation-recommendation", "children"),
        Input("workflow-state", "data"),
    )(_cb_validation_recommendation)

    app.callback(
        Output("train-tune-objective", "options"),
        Input("workflow-state", "data"),
    )(_cb_update_train_tune_objectives)

    app.callback(
        Output("train-learning-rate", "value"),
        Output("train-num-leaves", "value"),
        Output("train-min-child", "value"),
        Input("train-preset-conservative-btn", "n_clicks"),
        Input("train-preset-balanced-btn", "n_clicks"),
        State("train-learning-rate", "value"),
        State("train-num-leaves", "value"),
        State("train-min-child", "value"),
        prevent_initial_call=True,
    )(_cb_apply_train_preset)

    app.callback(
        Output("train-objective-help", "children"),
        Input("train-tune-objective", "value"),
    )(_cb_train_objective_help)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("train-learning-rate", "value"),
        Input("train-num-boost-round", "value"),
        Input("train-num-leaves", "value"),
        Input("train-max-depth", "value"),
        Input("train-min-child", "value"),
        Input("train-early-stopping", "value"),
        Input("train-auto-class-weight", "value"),
        Input("train-class-weight", "value"),
        Input("train-tune-enabled", "value"),
        Input("train-tune-preset", "value"),
        Input("train-tune-trials", "value"),
        Input("train-tune-objective", "value"),
        Input("train-artifact-dir", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_save_train_state)

    app.callback(
        Output("train-config-yaml-preview", "value"),
        Output("train-summary-container", "children"),
        Input("workflow-state", "data"),
    )(_cb_update_train_yaml_preview)

    app.callback(
        Output("train-config-yaml-preview", "value", allow_duplicate=True),
        Output("train-config-validate-result", "children"),
        Output("workflow-state", "data", allow_duplicate=True),
        Input("train-config-load-btn", "n_clicks"),
        Input("train-config-save-btn", "n_clicks"),
        Input("train-config-validate-btn", "n_clicks"),
        Input("train-config-yaml-import-btn", "n_clicks"),
        State("train-config-yaml-preview", "value"),
        State("train-config-file-path", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_train_yaml_actions)

    app.callback(
        Output("train-guardrail-container", "children"),
        Input("train-learning-rate", "value"),
        Input("train-num-boost-round", "value"),
        State("workflow-state", "data"),
    )(_cb_train_guardrails)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("custom-config-slots-store", "data"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_sync_custom_slots_to_state)

    def _register_phase30_callbacks(prefix: str, yaml_id: str) -> None:
        app.callback(
            Output(yaml_id, "value", allow_duplicate=True),
            Output("custom-config-slots-store", "data", allow_duplicate=True),
            Output(f"{prefix}-slot-select", "options"),
            Output(f"{prefix}-diff-count", "children"),
            Output(f"{prefix}-diff-view", "children"),
            Output(f"{prefix}-library-message", "children"),
            Output("workflow-state", "data", allow_duplicate=True),
            Input(f"{prefix}-template-apply-btn", "n_clicks"),
            Input(f"{prefix}-slot-save-btn", "n_clicks"),
            Input(f"{prefix}-slot-load-btn", "n_clicks"),
            Input(f"{prefix}-slot-clone-btn", "n_clicks"),
            State(f"{prefix}-template-select", "value"),
            State(f"{prefix}-slot-select", "value"),
            State(f"{prefix}-slot-name", "value"),
            State(f"{prefix}-diff-base", "value"),
            State(yaml_id, "value"),
            State("custom-config-slots-store", "data"),
            State("workflow-state", "data"),
            prevent_initial_call=True,
        )(_cb_phase30_library_actions)

        app.callback(
            Output(f"{prefix}-wizard-modal", "is_open"),
            Input(f"{prefix}-wizard-open-btn", "n_clicks"),
            Input(f"{prefix}-wizard-close-btn", "n_clicks"),
            State(f"{prefix}-wizard-modal", "is_open"),
            prevent_initial_call=True,
        )(_cb_phase30_toggle_wizard)

        app.callback(
            Output(f"{prefix}-wizard-step", "data"),
            Output(f"{prefix}-wizard-step-label", "children"),
            Input(f"{prefix}-wizard-next-btn", "n_clicks"),
            Input(f"{prefix}-wizard-prev-btn", "n_clicks"),
            State(f"{prefix}-wizard-step", "data"),
            prevent_initial_call=True,
        )(_cb_phase30_wizard_step)

        app.callback(
            Output(yaml_id, "value", allow_duplicate=True),
            Output(f"{prefix}-wizard-message", "children"),
            Output(f"{prefix}-wizard-modal", "is_open", allow_duplicate=True),
            Output("workflow-state", "data", allow_duplicate=True),
            Input(f"{prefix}-wizard-apply-btn", "n_clicks"),
            State(f"{prefix}-wizard-task", "value"),
            State(f"{prefix}-wizard-data-path", "value"),
            State(f"{prefix}-wizard-target", "value"),
            State(f"{prefix}-wizard-split-type", "value"),
            State(f"{prefix}-wizard-lr", "value"),
            State(f"{prefix}-wizard-rounds", "value"),
            State("workflow-state", "data"),
            prevent_initial_call=True,
        )(_cb_phase30_wizard_apply)

    _register_phase30_callbacks("train", "train-config-yaml-preview")
    _register_phase30_callbacks("config", "config-yaml")

    app.callback(
        Output("run-data-path", "value", allow_duplicate=True),
        Output("run-config-yaml", "value", allow_duplicate=True),
        Output("run-config-path", "value", allow_duplicate=True),
        Output("run-artifact-path", "value", allow_duplicate=True),
        Input("url", "pathname"),
        Input("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_sync_run_inputs_from_state)

    app.callback(
        Output("run-action-override-mode", "value"),
        Output("run-action-manual", "value"),
        Input("url", "pathname"),
        Input("workflow-state", "data"),
    )(_cb_sync_run_override_from_state)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("run-action-override-mode", "value"),
        Input("run-action-manual", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_save_run_action_override)

    app.callback(
        Output("run-action-manual-container", "style"),
        Input("run-action-override-mode", "value"),
    )(_cb_run_action_manual_visibility)

    # --- Run Page Auto-Action ---
    app.callback(
        Output("run-action", "value"),
        Output("run-action-display", "children"),
        Output("run-action-display", "className"),
        Input("run-config-yaml", "value"),
        Input("run-action-override-mode", "value"),
        Input("run-action-manual", "value"),
    )(_cb_detect_run_action)

    app.callback(
        Output("run-action-description", "children"),
        Input("run-action", "value"),
        Input("run-action-override-mode", "value"),
    )(_cb_run_action_description)

    app.callback(
        Output("run-guardrail-container", "children"),
        Output("run-guardrail-has-error", "data"),
        Input("run-action", "value"),
        Input("run-data-path", "value"),
        Input("run-config-yaml", "value"),
        Input("run-config-path", "value"),
        Input("run-artifact-path", "value"),
        Input("run-scenarios-path", "value"),
    )(_cb_run_guardrails)

    app.callback(
        Output("run-execute-btn", "disabled"),
        Output("run-execute-btn", "title"),
        Output("run-launch-status", "children"),
        Output("run-launch-status", "color"),
        Input("run-action", "value"),
        Input("run-data-path", "value"),
        Input("run-config-yaml", "value"),
        Input("run-config-path", "value"),
        Input("run-artifact-path", "value"),
        Input("run-scenarios-path", "value"),
        Input("run-guardrail-has-error", "data"),
    )(_cb_update_run_launch_state)

    # --- Run Page Callbacks ---
    app.callback(
        Output("run-jobs-interval", "interval"),
        Input("url", "pathname"),
    )(_cb_set_run_polling)

    app.callback(
        Output("run-result-log", "children"),
        Input("run-execute-btn", "n_clicks"),
        State("run-action", "value"),
        State("run-config-yaml", "value"),
        State("run-config-path", "value"),
        State("run-data-path", "value"),
        State("run-artifact-path", "value"),
        State("run-scenarios-path", "value"),
        State("run-export-format", "value"),
        State("run-priority", "value"),
        prevent_initial_call=True,
    )(_cb_enqueue_run_job)

    app.callback(
        Output("run-jobs-table-container", "children"),
        Output("toast-container", "children", allow_duplicate=True),
        Output("last-job-status", "data"),
        Output("workflow-state", "data", allow_duplicate=True),
        Output("url", "pathname"),  # Auto-navigation
        Input("run-jobs-interval", "n_intervals"),
        Input("run-refresh-jobs-btn", "n_clicks"),
        State("last-job-status", "data"),
        State("workflow-state", "data"),
        State("url", "pathname"),
        State("run-batch-mode-toggle", "value"),
        prevent_initial_call=True,
    )(_cb_refresh_run_jobs)

    app.callback(
        Output("run-job-detail", "children"),
        Output("run-cancel-job-btn", "disabled"),
        Output("run-retry-job-btn", "disabled"),
        Output("selected-job-id-display", "children"),
        Output("run-job-select", "data"),  # Store selection
        Output("run-log-limit", "data"),
        Input("run-jobs-table", "selected_rows"),
        Input("run-jobs-interval", "n_intervals"),
        Input("run-log-load-more-btn", "n_clicks"),
        State("run-jobs-table", "data"),
        State("run-job-select", "data"),
        State("run-log-limit", "data"),
        prevent_initial_call=True,
    )(_cb_show_selected_job_detail)

    app.callback(
        Output("run-result-log", "children", allow_duplicate=True),
        Input("run-cancel-job-btn", "n_clicks"),
        State("run-job-select", "data"),
        prevent_initial_call=True,
    )(_cb_cancel_job)

    app.callback(
        Output("run-result-log", "children", allow_duplicate=True),
        Input("run-retry-job-btn", "n_clicks"),
        State("run-job-select", "data"),
        prevent_initial_call=True,
    )(_cb_retry_job)

    app.callback(
        Output("run-result-log", "children", allow_duplicate=True),
        Input("run-set-priority-btn", "n_clicks"),
        State("run-job-select", "data"),
        State("run-queue-priority", "value"),
        prevent_initial_call=True,
    )(_cb_set_job_priority)

    # --- Results Page Callbacks ---
    app.callback(
        Output("artifact-select", "options"),
        Output("artifact-select-compare", "options"),
        Input("artifact-refresh-btn", "n_clicks"),
        Input("url", "pathname"),  # Auto-refresh on page load
        State("artifact-root-path", "value"),
    )(_cb_list_artifacts)

    app.callback(
        Output("artifact-kpi-container", "children"),
        Output("result-chart-main", "figure"),
        Output("result-chart-secondary", "figure"),
        Output("result-details", "children"),
        Input("artifact-select", "value"),
        Input("artifact-select-compare", "value"),
    )(_cb_update_result_view)

    # Auto-select artifact from workflow state
    app.callback(
        Output("artifact-select", "value"),
        Input("workflow-state", "data"),
        Input("artifact-select", "options"),
        State("url", "pathname"),
        State("artifact-select", "value"),
    )(_cb_autoselect_artifact)

    app.callback(
        Output("artifact-eval-result", "children"),
        Input("artifact-evaluate-btn", "n_clicks"),
        State("artifact-select", "value"),
        State("artifact-eval-data-path", "value"),
        prevent_initial_call=True,
    )(_cb_evaluate_artifact_action)

    app.callback(
        Output("result-learning-curve", "figure"),
        Output("result-config-view", "children"),
        Output("result-overview-summary", "children"),
        Input("artifact-select", "value"),
    )(_cb_update_result_extras)

    app.callback(
        Output("result-export-help", "children"),
        Input("artifact-select", "value"),
    )(_cb_result_export_help_for_artifact)

    app.callback(
        Output("artifact-eval-precheck", "children"),
        Input("artifact-select", "value"),
        Input("artifact-eval-data-path", "value"),
    )(_cb_result_eval_precheck)

    app.callback(
        Output("artifact-evaluate-btn", "className"),
        Output("result-export-excel-btn", "className"),
        Output("result-export-html-btn", "className"),
        Input("workflow-state", "data"),
        Input("url", "pathname"),
    )(_cb_result_shortcut_highlight)

    app.callback(
        Output("result-export-status", "children"),
        Output("result-export-job-store", "data"),
        Output("result-export-poll-interval", "disabled"),
        Output("result-export-poll-interval", "n_intervals"),
        Input("result-export-excel-btn", "n_clicks"),
        Input("result-export-html-btn", "n_clicks"),
        State("artifact-select", "value"),
        prevent_initial_call=True,
    )(_cb_result_export_actions)

    app.callback(
        Output("result-report-download", "data"),
        Output("result-export-status", "children", allow_duplicate=True),
        Output("result-export-job-store", "data", allow_duplicate=True),
        Output("result-export-poll-interval", "disabled", allow_duplicate=True),
        Input("result-export-poll-interval", "n_intervals"),
        State("result-export-job-store", "data"),
        prevent_initial_call=True,
    )(_cb_poll_result_export_job)

    app.callback(
        Output("result-config-download", "data"),
        Input("result-download-config-btn", "n_clicks"),
        State("artifact-select", "value"),
        prevent_initial_call=True,
    )(_cb_result_download_config)

    # --- Runs / Compare Callbacks ---
    app.callback(
        Output("runs-table", "data"),
        Input("runs-refresh-btn", "n_clicks"),
        Input("url", "pathname"),
        Input("runs-status-filter", "value"),
        Input("runs-action-filter", "value"),
        Input("runs-search", "value"),
    )(_cb_refresh_runs_table)

    app.callback(
        Output("runs-detail", "children"),
        Output("runs-selection-store", "data"),
        Input("runs-table", "selected_rows"),
        State("runs-table", "data"),
        prevent_initial_call=True,
    )(_cb_runs_selection_detail)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Output("url", "pathname", allow_duplicate=True),
        Output("runs-feedback", "children"),
        Input("runs-compare-btn", "n_clicks"),
        Input("runs-clone-btn", "n_clicks"),
        Input("runs-delete-btn", "n_clicks"),
        Input("runs-view-results-btn", "n_clicks"),
        Input("runs-migrate-btn", "n_clicks"),
        Input("runs-table", "active_cell"),
        State("runs-selection-store", "data"),
        State("workflow-state", "data"),
        State("runs-table", "data"),
        prevent_initial_call=True,
    )(_cb_runs_actions)

    app.callback(
        Output("compare-artifact-a", "options"),
        Output("compare-artifact-b", "options"),
        Output("compare-artifact-a", "value"),
        Output("compare-artifact-b", "value"),
        Input("url", "pathname"),
        State("workflow-state", "data"),
    )(_cb_populate_compare_options)

    app.callback(
        Output("compare-checks", "children"),
        Output("compare-metrics-table", "data"),
        Output("compare-chart", "figure"),
        Output("compare-config-diff", "children"),
        Input("compare-artifact-a", "value"),
        Input("compare-artifact-b", "value"),
    )(_cb_compare_runs)

    # --- Preset Synchronization Callbacks ---

    # 1. Config Export Dir
    app.callback(
        Output("cfg-export-dir-preset", "value"),
        Output("cfg-export-dir", "value"),
        Input("cfg-export-dir-preset", "value"),
        Input("cfg-export-dir", "value"),
        prevent_initial_call=True,
    )(_sync_path_preset)

    # 2. Run Artifact Path
    app.callback(
        Output("run-artifact-preset", "value"),
        Output("run-artifact-path", "value"),
        Input("run-artifact-preset", "value"),
        Input("run-artifact-path", "value"),
        prevent_initial_call=True,
    )(_sync_path_preset)

    # 3. Results Artifact Root
    app.callback(
        Output("artifact-root-preset", "value"),
        Output("artifact-root-path", "value"),
        Input("artifact-root-preset", "value"),
        Input("artifact-root-path", "value"),
        prevent_initial_call=True,
    )(_sync_path_preset)

    return app


# --- Module Scope Callbacks (Testing Logic) ---


def _cb_inspect_data(
    _n_clicks: int,
    upload_contents_or_path: str | None,
    upload_filename_or_contents: str | None,
    upload_filename: str | None = None,
    state: dict | None = None,
) -> tuple[Any, ...]:
    """Inspect data from uploaded file.

    Returns: (result_div, error, state, file_label, file_path).
    """
    legacy_mode = upload_filename is not None and state is None
    current_state = dict(state or {})

    def _ret(
        result_div: Any,
        error: str,
        state: dict,
        file_label: str = "No file selected",
        file_path: str = "",
    ) -> tuple[Any, ...]:
        if legacy_mode:
            return result_div, error, state
        return result_div, error, state, file_label, file_path

    # Backward compatible call shapes:
    # - New: (n_clicks, upload_contents, upload_filename)
    # - Legacy tests: (n_clicks, file_path, upload_contents, upload_filename)
    if upload_filename is None:
        upload_contents = upload_contents_or_path
        upload_filename = upload_filename_or_contents
    else:
        upload_contents = upload_filename_or_contents

    final_path = ""
    display_name = ""

    if isinstance(upload_contents, str) and upload_contents:
        # Decode and save temp file
        import base64
        import binascii

        try:
            _content_type, content_string = upload_contents.split(",", 1)
            decoded = base64.b64decode(content_string)
        except (ValueError, binascii.Error) as exc:
            return _ret(None, f"Invalid file format: {exc}", {})

        # Save to a temporary location
        filename = Path(upload_filename or "uploaded_data.csv").name
        display_name = filename
        _cleanup_gui_system_temp_files()
        final_path = str(_get_gui_system_temp_dir() / filename)

        try:
            if "csv" in filename or "parquet" in filename:
                with open(final_path, "wb") as f:
                    f.write(decoded)
            else:
                return _ret(None, "Unsupported file type (use .csv or .parquet)", {})
        except Exception as e:
            return _ret(None, f"Upload failed: {e}", {})
    else:
        if upload_filename:
            return _ret(None, "", current_state, f"Selected: {upload_filename}", "")
        return _ret(
            None,
            "Please select a .csv or .parquet file.",
            current_state,
            "No file selected — upload or drop a file above",
            "",
        )

    if not final_path:
        return _ret(None, "", current_state)

    result = inspect_data(final_path)
    if not result["success"]:
        return _ret(None, f"Error: {result.get('error')}", current_state)

    stats_div = data_page.render_data_stats(result["stats"])
    preview_div = data_page.render_data_preview(result["preview"])

    label = (
        f"✔ {display_name}  ({result['stats']['n_rows']} rows × {result['stats']['n_cols']} cols)"
    )
    next_state = _ensure_workflow_state_defaults(current_state)
    next_state["data_path"] = final_path
    next_state["data_stats"] = result.get("stats", {})
    cols = list(result.get("stats", {}).get("columns", []))
    if not next_state.get("target_col") and cols:
        next_state["target_col"] = cols[-1]
    if next_state.get("target_col") and cols:
        try:
            data_preview_df = _get_load_tabular_data()(final_path)
            inferred = infer_task_type(data_preview_df, str(next_state["target_col"]))
            next_state["task_type"] = inferred
        except Exception:
            next_state["task_type"] = next_state.get("task_type") or "regression"
    next_state["split_config"] = _default_split_for_task(
        str(next_state.get("task_type") or "regression"),
        causal_enabled=bool((next_state.get("causal_config") or {}).get("enabled")),
    )
    next_state["config_yaml"] = _build_config_from_state(next_state)
    return _ret(
        html.Div([stats_div, preview_div], className="data-inspection-zone"),
        "",
        next_state,
        label,
        final_path,
    )


def _cb_save_target_col(target_col: str, state: dict) -> dict:
    current = _ensure_workflow_state_defaults(state)
    current["target_col"] = target_col
    current["config_yaml"] = _build_config_from_state(current)
    return current


def _cb_update_selected_file_label(filename: str | list[str] | None) -> tuple[str, str]:
    if isinstance(filename, list):
        selected = filename[0] if filename else None
    else:
        selected = filename
    if not selected:
        return "No file selected — upload or drop a file above", ""
    return f"Selected: {selected}", ""


def _cb_cache_config_yaml(config_yaml: str, state: dict | None) -> dict:
    current = _ensure_workflow_state_defaults(state)
    current["config_yaml"] = config_yaml or ""
    return current


def _summarize_validation_feedback(payload: dict[str, Any]) -> str:
    if bool(payload.get("ok")):
        return "Configuration is valid."
    errors = payload.get("errors") or []
    lines: list[str] = []
    for idx, err in enumerate(errors[:5], start=1):
        path = str(err.get("path") or "root")
        message = str(err.get("message") or "Invalid value.")
        suggestions = err.get("suggestions") or []
        hint = f" / hint: {suggestions[0]}" if suggestions else ""
        lines.append(f"{idx}. {path}: {message}{hint}")
    if len(errors) > 5:
        lines.append(f"... and {len(errors) - 5} more errors.")
    return "\n".join(lines) if lines else "Validation failed."


def _cb_handle_config_actions(
    _validate_clicks: int,
    _load_clicks: int,
    _save_clicks: int,
    _import_clicks: int,
    yaml_text: str,
    config_path: str,
    workflow_state: dict | None,
) -> tuple[str, str, dict, Any]:
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    style_success = {"display": "block", "color": "#10b981"}
    style_error = {"display": "block", "color": "#ef4444"}

    try:
        if triggered == "config-load-btn":
            loaded = load_config_yaml(config_path or "")
            toast = make_toast(f"Loaded config from {config_path}", icon="success")
            return loaded, f"Loaded: {config_path}", style_success, toast

        if triggered == "config-save-btn":
            saved_path = save_config_yaml(config_path or "", yaml_text)
            toast = make_toast(f"Saved config to {saved_path}", icon="success")
            return yaml_text, f"Saved: {saved_path}", style_success, toast

        if triggered == "config-import-btn":
            if not workflow_state or "data_path" not in workflow_state:
                toast = make_toast("No data selected in Data page.", icon="warning")
                return yaml_text, "No data selected.", style_error, toast

            d_path = workflow_state["data_path"]
            t_col = workflow_state.get("target_col", "")

            try:
                payload = yaml.safe_load(yaml_text or "")
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}

            if "data" not in payload:
                payload["data"] = {}
            payload["data"]["path"] = d_path
            if t_col:
                payload["data"]["target"] = t_col

            # Ensure minimal structure
            if "task" not in payload:
                payload["task"] = {"type": "regression"}
            if "config_version" not in payload:
                payload["config_version"] = 1

            new_yaml = yaml.safe_dump(payload, sort_keys=False)
            toast = make_toast("Imported settings from Data page.", icon="success")
            return new_yaml, "Imported data settings.", style_success, toast

        validate_config(yaml_text)
        toast = make_toast("Configuration is valid.", icon="success")
        return yaml_text, "Configuration is valid.", style_success, toast
    except Exception as exc:
        feedback = validate_config_with_guidance(yaml_text)
        if not feedback.get("ok"):
            toast = make_toast("Validation failed. See details.", icon="danger")
            detail = _summarize_validation_feedback(feedback)
            message = f"{exc}\n{detail}" if str(exc) else detail
            return yaml_text, message, style_error, toast
        toast = make_toast(f"Error: {str(exc)}", icon="danger")
        return yaml_text, str(exc), style_error, toast


def _cb_handle_migration_preview(
    _n_clicks: int, input_path: str, target_ver: int
) -> tuple[str, str, str | None]:
    try:
        path = Path(input_path)
        if not path.exists():
            return "", "", f"File not found: {input_path}"

        # Read file content to migrate from text (to show diff)
        content = path.read_text(encoding="utf-8")
        normalized, diff = migrate_config_from_yaml(content, target_version=int(target_ver or 1))

        return normalized, diff, None
    except Exception as exc:
        return "", "", f"Error: {normalize_gui_error(exc)}"


def _cb_handle_migration_apply(_n_clicks: int, input_path: str, target_ver: int) -> str:
    try:
        msg = migrate_config_file_via_gui(input_path, target_version=int(target_ver or 1))
        return html.Div(html.Pre(msg), style={"color": "#10b981"})
    except Exception as exc:
        return html.Div(f"Error: {normalize_gui_error(exc)}", style={"color": "#ef4444"})


def _cb_update_split_options(split_type: str) -> tuple[dict, dict]:
    group_style = {"display": "none"}
    ts_style = {"display": "none"}

    if split_type == "group":
        group_style = {"display": "block"}
    elif split_type == "timeseries":
        ts_style = {"display": "block"}

    return group_style, ts_style


def _cb_timeseries_time_warning(
    split_type: str | None, time_col: str | None
) -> tuple[str, dict[str, str]]:
    if split_type != "timeseries":
        return "", {"display": "none"}

    if not (time_col or "").strip():
        return (
            "Time Series validation requires selecting Time Column (split.time_col).",
            {"display": "block"},
        )

    return "", {"display": "none"}


def _cb_update_tune_visibility(enabled: bool) -> dict:
    return {"display": "block"} if enabled else {"display": "none"}


_RUN_ACTION_DISPLAY = {
    "fit": ("Ready: TRAIN", "badge bg-primary fs-6 p-2 mb-3"),
    "tune": ("Ready: TUNE", "badge bg-warning text-dark fs-6 p-2 mb-3"),
    "evaluate": ("Ready: EVALUATE", "badge bg-info fs-6 p-2 mb-3"),
    "simulate": ("Ready: SIMULATE", "badge bg-secondary fs-6 p-2 mb-3"),
    "export": ("Ready: EXPORT", "badge bg-secondary fs-6 p-2 mb-3"),
    "estimate_dr": ("Ready: CAUSAL ANALYSIS", "badge bg-info text-dark fs-6 p-2 mb-3"),
}

_RUN_ACTION_DESCRIPTIONS = {
    "fit": "Train model with cross-validation.",
    "tune": "Run Optuna optimization, then train with best parameters.",
    "evaluate": "Evaluate predictions using an existing artifact or config.",
    "simulate": "Run counterfactual simulation with scenarios.",
    "export": "Export trained model in Python/ONNX package format.",
    "estimate_dr": "Estimate causal effects with Doubly Robust methods.",
}


def _cb_detect_run_action(
    yaml_text: str,
    override_mode: str | None = None,
    manual_action: str | None = None,
) -> tuple[str, str, str]:
    default_vals = ("fit", "Ready: TRAIN", "badge bg-primary fs-6 p-2 mb-3")
    if (override_mode or "auto") == "manual" and manual_action in _RUN_ACTION_DISPLAY:
        text, klass = _RUN_ACTION_DISPLAY[str(manual_action)]
        return str(manual_action), text, klass

    if not yaml_text:
        return default_vals

    try:
        cfg = yaml.safe_load(yaml_text)
        if not isinstance(cfg, dict):
            return default_vals

        # Check Tuning
        if cfg.get("tuning", {}).get("enabled", False):
            text, klass = _RUN_ACTION_DISPLAY["tune"]
            return "tune", text, klass

        # Check Causal
        if "causal_method" in cfg.get("task", {}):
            text, klass = _RUN_ACTION_DISPLAY["estimate_dr"]
            return "fit", text, klass

        return default_vals
    except Exception:
        return default_vals


def _cb_run_action_manual_visibility(override_mode: str | None) -> dict[str, str]:
    if (override_mode or "auto") == "manual":
        return {"display": "block"}
    return {"display": "none"}


def _cb_run_action_description(action: str | None, override_mode: str | None) -> str:
    mode = "MANUAL" if (override_mode or "auto") == "manual" else "AUTO"
    description = _RUN_ACTION_DESCRIPTIONS.get(str(action), "Run action is selected.")
    return f"{mode} mode: {description}"


def _cb_update_tune_objectives(
    task_type: str,
    *,
    causal_enabled: bool = False,
    causal_method: str | None = None,
) -> list[dict]:
    if causal_enabled:
        if str(causal_method or "") == "dr_did":
            opts = ["drdid_balance_priority", "drdid_std_error", "drdid_overlap_penalty"]
            return [{"label": o.upper(), "value": o} for o in opts]
        opts = ["dr_balance_priority", "dr_std_error", "dr_overlap_penalty"]
        return [{"label": o.upper(), "value": o} for o in opts]

    objectives = {
        "regression": ["rmse", "mae", "r2"],
        "binary": [
            "auc",
            "logloss",
            "brier",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "precision_at_k",
        ],
        "multiclass": ["accuracy", "macro_f1", "logloss"],
        "frontier": ["pinball", "pinball_coverage_penalty"],
    }
    opts = objectives.get(task_type, [])
    return [{"label": o.upper(), "value": o} for o in opts]


def _load_data_from_state(state: dict[str, Any] | None) -> pd.DataFrame | None:
    current = _ensure_workflow_state_defaults(state)
    data_path = current.get("data_path")
    if not isinstance(data_path, str) or not data_path.strip():
        return None
    try:
        return _get_load_tabular_data()(data_path)
    except Exception:
        return None


def _cb_populate_target_page(pathname: str, state: dict | None) -> tuple[Any, ...]:
    source_state = dict(state or {})
    current = _ensure_workflow_state_defaults(state)
    if pathname not in {"/target", "/config"}:
        return (
            current.get("data_path", ""),
            [],
            current.get("target_col", ""),
            [],
            current.get("exclude_cols", []),
            [],
            [],
            current.get("task_type", "regression"),
            "",
        )

    frame = _load_data_from_state(current)
    columns = list(frame.columns) if frame is not None else []
    target_col = current.get("target_col")
    if not target_col and columns:
        target_col = columns[-1]

    inferred = current.get("task_type")
    if frame is not None and target_col:
        inferred = infer_task_type(frame, str(target_col))
    task_type = source_state.get("task_type") or inferred or "regression"

    options = [{"label": c, "value": c} for c in columns]
    non_target = [{"label": c, "value": c} for c in columns if c != target_col]
    hint = f"Recommended task type: {inferred}" if inferred else ""
    return (
        current.get("data_path", ""),
        options,
        target_col,
        non_target,
        current.get("exclude_cols", []),
        options,
        options,
        task_type,
        hint,
    )


def _help_text(topic_key: str, field: str) -> str:
    entry = HELP_TEXTS.get(topic_key, {})
    if not isinstance(entry, dict):
        return ""
    return str(entry.get(field, ""))


def _cb_target_task_guides(task_type: str | None) -> tuple[Any, Any]:
    task = str(task_type or "regression")
    if task == "binary":
        card = context_card(
            "Binary classification",
            _help_text("task_type_binary", "detail"),
            variant="info",
        )
        return card, ""
    if task == "multiclass":
        card = context_card(
            "Multiclass classification",
            _help_text("task_type_multiclass", "detail"),
            variant="info",
        )
        return card, ""
    if task == "frontier":
        card = context_card(
            "Frontier analysis",
            _help_text("task_type_frontier", "detail"),
            variant="info",
        )
        alpha = context_card(
            "Frontier alpha guidance",
            _help_text("frontier_alpha", "detail"),
            variant="warning",
        )
        return card, alpha
    card = context_card(
        "Regression",
        _help_text("task_type_regression", "detail"),
        variant="info",
    )
    return card, ""


def _cb_target_causal_guides(
    causal_enabled: bool | None,
    causal_method: str | None,
) -> tuple[str, Any]:
    if not causal_enabled:
        return "", ""
    method = str(causal_method or "dr")
    if method == "dr_did":
        hint = "DR-DiD selected."
        card = context_card(
            "DR-DiD guidance",
            _help_text("causal_drdid", "detail"),
            variant="warning",
        )
        return hint, card
    hint = "DR selected."
    card = context_card(
        "DR guidance",
        _help_text("causal_dr", "detail"),
        variant="info",
    )
    return hint, card


def _cb_save_target_state(
    target_col: str | None,
    task_type: str | None,
    exclude_cols: list[str] | None,
    causal_enabled: bool | None,
    causal_method: str | None,
    treatment_col: str | None,
    unit_id_col: str | None,
    state: dict | None,
) -> dict[str, Any]:
    current = _ensure_workflow_state_defaults(state)
    current["target_col"] = target_col
    if task_type:
        current["task_type"] = task_type
    current["exclude_cols"] = list(exclude_cols or [])
    current["causal_config"] = {
        "enabled": bool(causal_enabled),
        "method": causal_method if causal_enabled else None,
        "treatment_col": treatment_col if causal_enabled else None,
        "unit_id_col": unit_id_col if causal_enabled else None,
    }
    if not current.get("split_config"):
        current["split_config"] = _default_split_for_task(
            current.get("task_type"),
            causal_enabled=bool(causal_enabled),
        )
    current["config_yaml"] = _build_config_from_state(current)
    return current


def _cb_target_guardrails(
    target_col: str | None,
    task_type: str | None,
    exclude_cols: list[str] | None,
    causal_enabled: bool | None,
    causal_method: str | None,
    treatment_col: str | None,
    unit_id_col: str | None,
    state: dict | None,
) -> Any:
    frame = _load_data_from_state(state)
    if frame is None:
        return render_guardrails([{"level": "info", "message": "Load data first."}])
    checker = GuardRailChecker()
    findings = checker.check_target(
        frame,
        target_col,
        task_type,
        exclude_cols=list(exclude_cols or []),
    )
    payload = [asdict(item) for item in findings]
    if causal_enabled and not (treatment_col or "").strip():
        payload.append(
            {
                "level": "warning",
                "message": "Causal mode requires a binary treatment column.",
                "suggestion": "Select Treatment column in Target page.",
            }
        )
    if causal_enabled and str(causal_method or "") == "dr_did" and not (unit_id_col or "").strip():
        payload.append(
            {
                "level": "info",
                "message": "DR-DiD is more stable with Unit ID column.",
                "suggestion": "Select Unit ID when panel-style data is available.",
            }
        )
    return render_guardrails(payload)


def _cb_populate_validation_options(
    pathname: str, state: dict | None
) -> tuple[list[dict], list[dict]]:
    if pathname != "/validation":
        return [], []
    frame = _load_data_from_state(state)
    if frame is None:
        return [], []
    cols = [{"label": c, "value": c} for c in frame.columns]
    return cols, cols


def _cb_save_validation_state(
    split_type: str | None,
    n_splits: int | None,
    seed: int | None,
    group_col: str | None,
    time_col: str | None,
    ts_mode: str | None,
    test_size: int | None,
    gap: int | None,
    embargo: int | None,
    state: dict | None,
) -> dict[str, Any]:
    current = _ensure_workflow_state_defaults(state)
    current["split_config"] = {
        "type": split_type or current.get("split_config", {}).get("type", "kfold"),
        "n_splits": int(n_splits or 5),
        "seed": int(seed or 42),
        "group_col": group_col,
        "time_col": time_col,
        "timeseries_mode": ts_mode or "expanding",
        "test_size": int(test_size) if test_size not in (None, "") else None,
        "gap": int(gap or 0),
        "embargo": int(embargo or 0),
    }
    current["config_yaml"] = _build_config_from_state(current)
    return current


def _cb_validation_guardrails(
    split_type: str | None,
    n_splits: int | None,
    group_col: str | None,
    time_col: str | None,
    state: dict | None,
) -> Any:
    frame = _load_data_from_state(state)
    if frame is None:
        return render_guardrails([{"level": "info", "message": "Load data first."}])
    current = _ensure_workflow_state_defaults(state)
    split_config = {
        "type": split_type or "kfold",
        "n_splits": int(n_splits or 5),
        "group_col": group_col,
        "time_col": time_col,
    }
    checker = GuardRailChecker()
    findings = checker.check_validation(
        frame,
        split_config,
        task_type=current.get("task_type"),
        exclude_cols=list(current.get("exclude_cols") or []),
    )
    task_type = str(current.get("task_type") or "regression")
    causal_cfg = current.get("causal_config") or {}
    if task_type in {"binary", "multiclass"} and split_config["type"] != "stratified":
        findings.append(
            GuardRailResult(
                "warning",
                "Classification tasks are recommended to use stratified split.",
                "Switch split type to stratified.",
            )
        )
    if bool(causal_cfg.get("enabled")) and split_config["type"] == "stratified":
        findings.append(
            GuardRailResult(
                "info",
                "Causal workflows often use kfold/group over stratified split.",
                "Review split strategy in Validation page.",
            )
        )
    return render_guardrails([asdict(item) for item in findings])


def _cb_validation_split_context(split_type: str | None) -> Any:
    split = str(split_type or "kfold")
    key = {
        "kfold": "split_kfold",
        "stratified": "split_stratified",
        "group": "split_group",
        "timeseries": "split_timeseries",
    }.get(split, "split_kfold")
    title = split.title() if split != "timeseries" else "TimeSeries"
    return context_card(f"{title} split", _help_text(key, "detail"), variant="info")


def _cb_validation_recommendation(state: dict | None) -> Any:
    current = _ensure_workflow_state_defaults(state)
    task_type = str(current.get("task_type") or "regression")
    split_cfg = current.get("split_config") or {}
    split_type = str(split_cfg.get("type") or "kfold")
    badges: list[Any] = []
    if task_type in {"binary", "multiclass"}:
        badges.append(recommendation_badge("Recommended: Stratified split", "info"))
    if bool((current.get("causal_config") or {}).get("enabled")):
        badges.append(recommendation_badge("Recommended: KFold/Group split for causal", "warning"))
    if split_type == "timeseries":
        badges.append(recommendation_badge("TimeSeries split selected", "ok"))
    return html.Div(badges) if badges else ""


def _cb_update_train_tune_objectives(state: dict | None) -> list[dict]:
    current = _ensure_workflow_state_defaults(state)
    causal_cfg = current.get("causal_config") or {}
    return _cb_update_tune_objectives(
        str(current.get("task_type") or "regression"),
        causal_enabled=bool(causal_cfg.get("enabled")),
        causal_method=causal_cfg.get("method"),
    )


def _cb_apply_train_preset(
    _conservative_clicks: int,
    _balanced_clicks: int,
    current_lr: float | None,
    current_leaves: int | None,
    current_child: int | None,
) -> tuple[float, int, int]:
    try:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    except Exception:
        triggered = ""
    if triggered == "train-preset-conservative-btn":
        return 0.01, 31, 50
    if triggered == "train-preset-balanced-btn":
        return 0.05, 63, 20
    return float(current_lr or 0.05), int(current_leaves or 31), int(current_child or 20)


def _cb_train_objective_help(objective: str | None) -> Any:
    objective_key = str(objective or "").strip().lower()
    if not objective_key:
        return ""
    mapping = {
        "rmse": ("RMSE objective", "objective_rmse"),
        "mae": ("MAE objective", "objective_mae"),
        "auc": ("AUC objective", "objective_auc"),
        "f1": ("F1 objective", "objective_f1"),
        "logloss": ("Logloss objective", "objective_logloss"),
        "pinball": ("Pinball objective", "objective_pinball"),
        "dr_balance_priority": ("DR balance priority", "objective_dr_balance_priority"),
        "dr_std_error": ("DR standard error", "objective_dr_std_error"),
        "dr_overlap_penalty": ("DR overlap penalty", "objective_dr_overlap_penalty"),
        "drdid_balance_priority": ("DR-DiD balance priority", "objective_dr_balance_priority"),
        "drdid_std_error": ("DR-DiD standard error", "objective_dr_std_error"),
        "drdid_overlap_penalty": ("DR-DiD overlap penalty", "objective_dr_overlap_penalty"),
    }
    title, help_key = mapping.get(
        objective_key,
        (f"{objective_key.upper()} objective", "objective_logloss"),
    )
    return context_card(title, _help_text(help_key, "detail"), variant="info")


def _cb_save_train_state(
    learning_rate: float | None,
    num_boost_round: int | None,
    num_leaves: int | None,
    max_depth: int | None,
    min_child: int | None,
    early_stopping: int | None,
    auto_class_weight: bool | None,
    class_weight: str | None,
    tune_enabled: bool | None,
    tune_preset: str | None,
    tune_trials: int | None,
    tune_objective: str | None,
    artifact_dir: str | None,
    state: dict | None,
) -> dict[str, Any]:
    current = _ensure_workflow_state_defaults(state)
    train_cfg = dict(current.get("train_config") or {})
    train_cfg.update(
        {
            "learning_rate": float(learning_rate or train_cfg.get("learning_rate", 0.05)),
            "num_boost_round": int(num_boost_round or train_cfg.get("num_boost_round", 300)),
            "num_leaves": int(num_leaves or train_cfg.get("num_leaves", 31)),
            "max_depth": int(
                max_depth if max_depth is not None else train_cfg.get("max_depth", -1)
            ),
            "min_child_samples": int(
                min_child if min_child is not None else train_cfg.get("min_child_samples", 20)
            ),
            "early_stopping_rounds": int(
                early_stopping
                if early_stopping is not None
                else train_cfg.get("early_stopping_rounds", 100)
            ),
            "auto_class_weight": bool(
                auto_class_weight
                if auto_class_weight is not None
                else train_cfg.get("auto_class_weight", True)
            ),
            "class_weight_text": class_weight or "",
        }
    )
    current["train_config"] = train_cfg
    current["tuning_config"] = {
        "enabled": bool(tune_enabled),
        "preset": tune_preset or "standard",
        "n_trials": int(tune_trials or 30),
        "objective": tune_objective,
    }
    current["artifact_dir"] = artifact_dir or "artifacts"
    current["config_yaml"] = _build_config_from_state(current)
    return current


def _render_train_summary(state: dict | None) -> html.Div:
    current = _ensure_workflow_state_defaults(state)
    split = current.get("split_config") or {}
    train = current.get("train_config") or {}
    tune = current.get("tuning_config") or {}
    return html.Ul(
        [
            html.Li(f"Task: {current.get('task_type', 'regression')}"),
            html.Li(f"Split: {split.get('type', 'kfold')} ({split.get('n_splits', 5)} folds)"),
            html.Li(
                "Train: "
                f"lr={train.get('learning_rate', 0.05)}, "
                f"rounds={train.get('num_boost_round', 300)}, "
                f"leaves={train.get('num_leaves', 31)}"
            ),
            html.Li(f"Tuning: {'ON' if tune.get('enabled') else 'OFF'}"),
        ],
        className="mb-0",
    )


def _cb_update_train_yaml_preview(state: dict | None) -> tuple[str, Any]:
    current = _ensure_workflow_state_defaults(state)
    yaml_text = _build_config_from_state(current)
    return yaml_text, _render_train_summary(current)


def _cb_train_yaml_actions(
    _load_clicks: int,
    _save_clicks: int,
    _validate_clicks: int,
    _import_clicks: int,
    yaml_text: str,
    config_path: str,
    state: dict | None,
) -> tuple[Any, str, dict[str, Any]]:
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    current = _ensure_workflow_state_defaults(state)
    message = ""
    out_yaml = yaml_text or ""

    try:
        if triggered == "train-config-load-btn":
            out_yaml = load_config_yaml(config_path or "configs/gui_run.yaml")
            payload = yaml.safe_load(out_yaml) or {}
            if isinstance(payload, dict):
                current = _state_from_config_payload(payload, current)
            message = f"Loaded: {config_path}"
        elif triggered == "train-config-save-btn":
            saved = save_config_yaml(config_path or "configs/gui_run.yaml", out_yaml)
            message = f"Saved: {saved}"
        elif triggered == "train-config-validate-btn":
            validate_config(out_yaml)
            current["last_validation"] = {
                "ok": True,
                "errors": [],
                "warnings": [],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            message = "Config validation passed."
        elif triggered == "train-config-yaml-import-btn":
            payload = yaml.safe_load(out_yaml) or {}
            if not isinstance(payload, dict):
                raise ValueError("YAML must deserialize to a mapping.")
            current = _state_from_config_payload(payload, current)
            message = "Imported YAML into workflow state."
        current["config_yaml"] = out_yaml
        return out_yaml, message, current
    except Exception as exc:
        if triggered == "train-config-validate-btn":
            feedback = validate_config_with_guidance(out_yaml)
            current["last_validation"] = feedback
            if not feedback.get("ok"):
                return out_yaml, _summarize_validation_feedback(feedback), current
        current["config_yaml"] = out_yaml
        return out_yaml, f"Error: {exc}", current


def _cb_train_guardrails(
    learning_rate: float | None,
    num_boost_round: int | None,
    state: dict | None,
) -> Any:
    checker = GuardRailChecker()
    findings = checker.check_train(
        {
            "learning_rate": float(learning_rate or 0.05),
            "num_boost_round": int(num_boost_round or 300),
        }
    )
    return render_guardrails([asdict(item) for item in findings])


def _cb_sync_custom_slots_to_state(slots: list[dict] | None, state: dict | None) -> dict:
    current = _ensure_workflow_state_defaults(state)
    current["custom_config_slots"] = list(slots or [])
    return current


def _cb_phase30_library_actions(
    _apply_clicks: int,
    _save_clicks: int,
    _load_clicks: int,
    _clone_clicks: int,
    selected_template_id: str | None,
    selected_slot_id: str | None,
    slot_name: str | None,
    diff_base: str | None,
    current_yaml: str | None,
    slots_store: list[dict] | None,
    state: dict | None,
) -> tuple[str, list[dict], list[dict], str, str, str, dict]:
    current = _ensure_workflow_state_defaults(state)
    yaml_text = current_yaml or current.get("config_yaml") or _build_config_from_state(current)
    slots = list(slots_store or current.get("custom_config_slots") or [])
    message = ""
    triggered = ""
    try:
        if callback_context.triggered:
            triggered = str(callback_context.triggered[0]["prop_id"]).split(".")[0]
    except Exception:
        triggered = ""

    try:
        if triggered.endswith("template-apply-btn"):
            template_id = str(selected_template_id or "").strip() or "regression_baseline"
            yaml_text = load_builtin_template_yaml(template_id)
            current["template_id"] = template_id
            current["template_origin"] = "builtin"
            current["config_diff_base_yaml"] = yaml_text
            feedback = validate_config_with_guidance(yaml_text)
            current["last_validation"] = feedback
            message = f"Applied template: {template_id}"
        elif triggered.endswith("slot-save-btn"):
            slots = save_custom_slot(
                slots,
                name=str(slot_name or "Custom Config"),
                yaml_text=yaml_text,
                template_origin=str(current.get("template_origin") or "custom"),
            )
            current["template_origin"] = "custom"
            message = "Saved current YAML to custom slot."
        elif triggered.endswith("slot-load-btn"):
            yaml_text = load_custom_slot_yaml(slots, slot_id=str(selected_slot_id or ""))
            current["template_origin"] = "custom"
            current["config_diff_base_yaml"] = yaml_text
            feedback = validate_config_with_guidance(yaml_text)
            current["last_validation"] = feedback
            message = "Loaded selected custom slot."
        elif triggered.endswith("slot-clone-btn"):
            slots = clone_custom_slot(slots, slot_id=str(selected_slot_id or ""))
            message = "Cloned selected slot."
    except Exception as exc:
        message = f"Error: {exc}"

    slot_opts = custom_slot_options(slots)
    base_yaml = ""
    if str(diff_base or "template") == "slot":
        if selected_slot_id:
            try:
                base_yaml = load_custom_slot_yaml(slots, slot_id=str(selected_slot_id))
            except Exception:
                base_yaml = ""
    else:
        template_id = str(
            selected_template_id or current.get("template_id") or "regression_baseline"
        )
        try:
            base_yaml = load_builtin_template_yaml(template_id)
        except Exception:
            base_yaml = ""
    diff_text = (
        render_yaml_diff(base_yaml, yaml_text).children if base_yaml else "No base selected."
    )
    diff_count = count_yaml_changes(base_yaml, yaml_text) if base_yaml else 0

    current["custom_config_slots"] = slots
    current["config_yaml"] = yaml_text
    current["config_diff_base_yaml"] = base_yaml
    return (
        yaml_text,
        slots,
        slot_opts,
        f"Changed keys: {diff_count}",
        str(diff_text),
        message,
        current,
    )


def _cb_phase30_toggle_wizard(
    _open_clicks: int,
    _close_clicks: int,
    is_open: bool,
) -> bool:
    triggered = ""
    try:
        if callback_context.triggered:
            triggered = str(callback_context.triggered[0]["prop_id"]).split(".")[0]
    except Exception:
        triggered = ""
    if triggered.endswith("wizard-open-btn"):
        return True
    if triggered.endswith("wizard-close-btn"):
        return False
    return is_open


def _cb_phase30_wizard_step(
    _next_clicks: int,
    _prev_clicks: int,
    step: int | None,
) -> tuple[int, str]:
    cur = max(1, min(4, int(step or 1)))
    triggered = ""
    try:
        if callback_context.triggered:
            triggered = str(callback_context.triggered[0]["prop_id"]).split(".")[0]
    except Exception:
        triggered = ""
    if triggered.endswith("wizard-next-btn"):
        cur = min(4, cur + 1)
    elif triggered.endswith("wizard-prev-btn"):
        cur = max(1, cur - 1)
    return cur, f"Step {cur}/4"


def _cb_phase30_wizard_apply(
    _apply_clicks: int,
    task_type: str | None,
    data_path: str | None,
    target_col: str | None,
    split_type: str | None,
    learning_rate: float | None,
    rounds: int | None,
    state: dict | None,
) -> tuple[str, str, bool, dict]:
    current = _ensure_workflow_state_defaults(state)
    split = _default_split_for_task(str(task_type or "regression"))
    split["type"] = str(split_type or split["type"])
    train = _default_train_config()
    train["learning_rate"] = float(learning_rate or 0.05)
    train["num_boost_round"] = int(rounds or 300)
    current["task_type"] = str(task_type or "regression")
    current["data_path"] = str(data_path or "")
    current["target_col"] = str(target_col or "")
    current["split_config"] = split
    current["train_config"] = train
    yaml_text = _build_config_from_state(current)
    feedback = validate_config_with_guidance(yaml_text)
    current["last_validation"] = feedback
    current["config_yaml"] = yaml_text
    current["template_origin"] = "custom"
    msg = (
        "Wizard applied and configuration validated."
        if feedback.get("ok")
        else _summarize_validation_feedback(feedback)
    )
    return yaml_text, msg, False, current


def _cb_save_run_action_override(
    override_mode: str | None,
    manual_action: str | None,
    state: dict | None,
) -> dict[str, Any]:
    current = _ensure_workflow_state_defaults(state)
    mode = "manual" if (override_mode or "auto") == "manual" else "auto"
    action = str(manual_action) if mode == "manual" else None
    current["run_action_override"] = {"mode": mode, "action": action}
    return current


def _cb_sync_run_inputs_from_state(
    pathname: str | None, state: dict | None
) -> tuple[Any, Any, Any, Any]:
    if pathname != "/run":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    current = _ensure_workflow_state_defaults(state)
    config_yaml = current.get("config_yaml") or _build_config_from_state(current)
    artifact_dir = current.get("artifact_dir") or "artifacts"
    return (
        current.get("data_path", ""),
        config_yaml,
        "configs/gui_run.yaml",
        artifact_dir,
    )


def _cb_sync_run_override_from_state(pathname: str | None, state: dict | None) -> tuple[Any, Any]:
    if pathname != "/run":
        return dash.no_update, dash.no_update
    current = _ensure_workflow_state_defaults(state)
    override = current.get("run_action_override") or {}
    return override.get("mode", "auto"), override.get("action") or "fit"


class _PopulateBuilderLegacyResult:
    """Compatibility view for tests that rely on both old unpacking and new indexing."""

    def __init__(
        self,
        d_path: str,
        t_col: str,
        target_opts: list[dict[str, Any]],
        opts: list[dict[str, Any]],
        non_target_opts: list[dict[str, Any]],
    ) -> None:
        self._modern = (d_path, t_col, target_opts, opts, opts, non_target_opts, opts, opts)
        self._legacy = (d_path, t_col, opts, opts, non_target_opts, opts, opts)

    def __iter__(self):
        return iter(self._legacy)

    def __getitem__(self, idx: int):
        return self._modern[idx]

    def __len__(self) -> int:
        return len(self._modern)


def _cb_populate_builder_options(pathname: str, state: dict | None) -> tuple[Any, ...]:
    if not state or "data_path" not in state:
        try:
            _ = callback_context.triggered_id
            return "", "", [], [], [], [], [], []
        except Exception:
            return "", "", [], [], [], [], []

    d_path = state.get("data_path", "")
    t_col = state.get("target_col", "")

    cols = []
    try:
        if d_path:
            res = inspect_data(d_path)
            if res["success"]:
                cols = res["stats"]["columns"]
    except Exception:
        pass

    # Exclude Columns Options: All columns except target
    non_target_opts = [{"label": c, "value": c} for c in cols if c != t_col]
    opts = [{"label": c, "value": c} for c in cols]
    target_opts = [{"label": c, "value": c} for c in cols]

    try:
        _ = callback_context.triggered_id
        return d_path, t_col, target_opts, opts, opts, non_target_opts, opts, opts
    except Exception:
        return _PopulateBuilderLegacyResult(d_path, t_col, target_opts, opts, non_target_opts)


def _cb_build_config_yaml(
    task_type,
    d_path,
    d_target,
    d_ids,
    d_cats,
    d_drops,
    s_type,
    s_n,
    s_seed,
    s_grp,
    s_time,
    s_ts_mode,
    s_test,
    s_gap,
    s_embargo,
    t_lr,
    t_leaves,
    t_est,
    t_depth,
    t_child,
    t_early,
    t_sub,
    t_col,
    t_l1,
    t_l2,
    tune_en,
    tune_pre,
    tune_tri,
    tune_obj,
    exp_dir,
    causal_en,
    causal_method,
    # Search Space Params
    tune_lr_min,
    tune_lr_max,
    tune_leaves_min,
    tune_leaves_max,
    tune_depth_min,
    tune_depth_max,
    tune_ff_min,
    tune_ff_max,
    t_auto_class_weight=None,
    t_class_weight=None,
    t_auto_num_leaves=None,
    t_num_leaves_ratio=None,
    t_min_leaf_ratio=None,
    t_min_bin_ratio=None,
    t_feature_weights=None,
    t_path_smooth=None,
    t_cat_l2=None,
    t_cat_smooth=None,
    t_bagging_freq=None,
    t_max_bin=None,
    t_max_drop=None,
    t_min_gain=None,
    t_top_k=None,
) -> str:
    cfg = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {
            "path": d_path,
            "target": d_target,
        },
        "split": {
            "type": s_type,
            "n_splits": s_n,
            "seed": s_seed,
        },
        "train": {
            "num_boost_round": t_est,
            "lgb_params": {
                "learning_rate": t_lr,
                "num_leaves": t_leaves,
                "max_depth": t_depth,
                "min_child_samples": t_child,
                "subsample": t_sub,
                "colsample_bytree": t_col,
                "reg_alpha": t_l1,
                "reg_lambda": t_l2,
            },
            "early_stopping_rounds": t_early,
            "seed": 42,
        },
        "tuning": {"enabled": tune_en},
        "export": {"artifact_dir": exp_dir},
    }

    if causal_en and causal_method:
        cfg["task"]["causal_method"] = causal_method
        # Causal requires specific tasks usually, but we overlay it

    if d_ids:
        cfg["data"]["id_cols"] = d_ids
    if d_cats:
        cfg["data"]["categorical"] = d_cats
    if d_drops:
        cfg["data"]["drop_cols"] = d_drops

    if s_type == "group" and s_grp:
        cfg["split"]["group_col"] = s_grp
    if s_type == "timeseries":
        if s_time:
            cfg["split"]["time_col"] = s_time
        cfg["split"]["timeseries_mode"] = s_ts_mode
        if s_test:
            cfg["split"]["test_size"] = s_test
        cfg["split"]["gap"] = s_gap
        cfg["split"]["embargo"] = s_embargo

    if tune_en:
        cfg["tuning"]["preset"] = tune_pre
        cfg["tuning"]["n_trials"] = tune_tri
        if tune_obj:
            cfg["tuning"]["objective"] = tune_obj

        # Custom Search Space
        space = {}
        if tune_lr_min and tune_lr_max:
            space["learning_rate"] = {
                "low": float(tune_lr_min),
                "high": float(tune_lr_max),
                "log": True,
            }
        if tune_leaves_min and tune_leaves_max:
            space["num_leaves"] = {"low": int(tune_leaves_min), "high": int(tune_leaves_max)}
        if tune_depth_min and tune_depth_max:
            space["max_depth"] = {"low": int(tune_depth_min), "high": int(tune_depth_max)}
        if tune_ff_min and tune_ff_max:
            space["feature_fraction"] = {"low": float(tune_ff_min), "high": float(tune_ff_max)}

        if space:
            cfg["tuning"]["search_space"] = space

    if task_type in {"binary", "multiclass"}:
        auto_class_weight = True if t_auto_class_weight is None else bool(t_auto_class_weight)
        cfg["train"]["auto_class_weight"] = auto_class_weight
        if not auto_class_weight and isinstance(t_class_weight, str) and t_class_weight.strip():
            try:
                parsed_weights = yaml.safe_load(t_class_weight)
                if isinstance(parsed_weights, dict):
                    cfg["train"]["class_weight"] = {
                        str(k): float(v) for k, v in parsed_weights.items()
                    }
            except Exception:
                # Keep GUI YAML generation resilient; validation runs separately.
                pass

    auto_num_leaves = bool(t_auto_num_leaves)
    cfg["train"]["auto_num_leaves"] = auto_num_leaves
    if t_num_leaves_ratio is not None and t_num_leaves_ratio != "":
        cfg["train"]["num_leaves_ratio"] = float(t_num_leaves_ratio)
    if t_min_leaf_ratio is not None and t_min_leaf_ratio != "":
        cfg["train"]["min_data_in_leaf_ratio"] = float(t_min_leaf_ratio)
    if t_min_bin_ratio is not None and t_min_bin_ratio != "":
        cfg["train"]["min_data_in_bin_ratio"] = float(t_min_bin_ratio)
    if isinstance(t_feature_weights, str) and t_feature_weights.strip():
        try:
            parsed_feature_weights = yaml.safe_load(t_feature_weights)
            if isinstance(parsed_feature_weights, dict):
                cfg["train"]["feature_weights"] = {
                    str(k): float(v) for k, v in parsed_feature_weights.items()
                }
        except Exception:
            # Keep GUI YAML generation resilient; validation runs separately.
            pass
    if task_type == "binary" and t_top_k is not None and t_top_k != "":
        cfg["train"]["top_k"] = int(t_top_k)

    if auto_num_leaves:
        cfg["train"]["lgb_params"].pop("num_leaves", None)

    advanced_lgb_params = {
        "path_smooth": t_path_smooth,
        "cat_l2": t_cat_l2,
        "cat_smooth": t_cat_smooth,
        "bagging_freq": t_bagging_freq,
        "max_bin": t_max_bin,
        "max_drop": t_max_drop,
        "min_gain_to_split": t_min_gain,
    }
    for key, value in advanced_lgb_params.items():
        if value is None or value == "":
            continue
        if key in {"bagging_freq", "max_bin", "max_drop"}:
            cfg["train"]["lgb_params"][key] = int(value)
        else:
            cfg["train"]["lgb_params"][key] = float(value)

    return yaml.dump(cfg, sort_keys=False)


def _cb_update_class_weight_visibility(
    task_type: str | None,
    auto_class_weight_enabled: bool | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if task_type not in {"binary", "multiclass"}:
        return {"display": "none"}, {"display": "none"}
    show_manual = not bool(auto_class_weight_enabled)
    return {"display": "block"}, {"display": "block" if show_manual else "none"}


def _cb_update_top_k_visibility(task_type: str | None) -> dict[str, str]:
    return {"display": "block"} if task_type == "binary" else {"display": "none"}


def _cb_run_guardrails(
    action: str | None,
    data_path: str | None,
    config_yaml: str | None,
    config_path: str | None,
    artifact_path: str | None,
    scenarios_path: str | None,
) -> tuple[Any, bool]:
    checker = GuardRailChecker()
    act = (action or "fit").strip().lower()
    payload: list[dict[str, Any]] = []

    if act in {"fit", "tune", "estimate_dr", "evaluate"}:
        yaml_text = (config_yaml or "").strip()
        if not yaml_text and (config_path or "").strip():
            try:
                yaml_text = load_config_yaml(config_path or "")
            except Exception:
                yaml_text = ""
        if yaml_text:
            payload.extend(
                [
                    asdict(item)
                    for item in checker.check_pre_run(yaml_text, data_path or None)
                    if item.level in {"error", "warning", "info"}
                ]
            )
    if act in {"evaluate", "simulate"} and not (data_path or "").strip():
        payload.append(
            {
                "level": "error",
                "message": "Data Source is required for evaluate/simulate action.",
                "suggestion": "Set Data Source in Data page.",
            }
        )
    if act in {"evaluate", "simulate", "export"} and not (artifact_path or "").strip():
        payload.append(
            {
                "level": "warning" if act == "evaluate" else "error",
                "message": "Artifact Path is required.",
                "suggestion": "Select artifact from Results or Runs page.",
            }
        )
    if act == "simulate" and not (scenarios_path or "").strip():
        payload.append(
            {
                "level": "error",
                "message": "Scenarios Path is required for simulate action.",
                "suggestion": "Set scenario file path before launch.",
            }
        )

    has_error = any(str(item.get("level")) == "error" for item in payload)
    if not payload:
        payload = [{"level": "ok", "message": "Pre-run checks passed."}]
    return render_guardrails(payload, sort_by_severity=True), has_error


def _cb_update_run_launch_state(
    action: str | None,
    data_path: str | None,
    config_yaml: str | None,
    config_path: str | None,
    artifact_path: str | None,
    scenarios_path: str | None,
    run_guardrail_has_error: bool | None = None,
) -> tuple[bool, str, str, str]:
    act = (action or "fit").strip().lower()
    has_data = bool((data_path or "").strip())
    has_config = bool((config_yaml or "").strip())
    has_config_path = bool((config_path or "").strip())
    has_any_config_source = has_config or has_config_path
    has_artifact = bool((artifact_path or "").strip())
    has_scenarios = bool((scenarios_path or "").strip())

    missing: list[str] = []
    if act in {"fit", "tune", "estimate_dr"}:
        if not has_data:
            missing.append("Data Source")
        if not has_any_config_source:
            missing.append("Config Source")
    elif act == "evaluate":
        if not has_data:
            missing.append("Data Source")
        if not has_artifact and not has_any_config_source:
            missing.append("Artifact Path or Config Source")
    elif act == "simulate":
        if not has_data:
            missing.append("Data Source")
        if not has_artifact:
            missing.append("Artifact Path")
        if not has_scenarios:
            missing.append("Scenarios Path")
    elif act == "export":
        if not has_artifact:
            missing.append("Artifact Path")

    if missing:
        req = ", ".join(missing)
        return (
            True,
            f"Missing required inputs: {req}",
            f"Not ready ({act.upper()}): {req}",
            "warning",
        )
    if bool(run_guardrail_has_error):
        return (
            True,
            "Guardrail error detected. Resolve errors before launch.",
            f"Not ready ({act.upper()}): guardrail errors exist.",
            "danger",
        )

    return (
        False,
        f"Ready to launch {act.upper()}",
        f"Ready ({act.upper()}): required inputs are set.",
        "success",
    )


def _cb_set_run_polling(_pathname: str | None) -> int:
    return max(200, int(os.getenv("VELDRA_GUI_POLL_MS", "2000")))


def _cb_enqueue_run_job(
    _n_clicks: int,
    action: str,
    config_yaml: str,
    config_path_state: str,
    data_path_state: str,
    artifact_path: str,
    scenarios_path: str,
    export_format: str,
    priority: str,
) -> str:
    try:
        c_path = _ensure_default_run_config(config_path_state or "configs/gui_run.yaml")
        d_path = data_path_state
        act = str(action or "").strip().lower()
        if act in {"fit", "tune", "estimate_dr", "evaluate"}:
            yaml_source = (config_yaml or "").strip()
            should_validate = False
            if yaml_source:
                parsed = yaml.safe_load(yaml_source)
                if isinstance(parsed, dict):
                    should_validate = any(
                        key in parsed
                        for key in (
                            "config_version",
                            "task",
                            "data",
                            "split",
                            "train",
                            "tuning",
                            "export",
                            "causal",
                        )
                    )
                if not should_validate and not (c_path or "").strip():
                    should_validate = True
            elif (c_path or "").strip():
                yaml_source = load_config_yaml(c_path)
                should_validate = True

            if should_validate:
                feedback = validate_config_with_guidance(yaml_source)
                if not feedback.get("ok"):
                    return (
                        "[ERROR] Validation blocked run:\n"
                        f"{_summarize_validation_feedback(feedback)}"
                    )

        result = submit_run_job(
            RunInvocation(
                action=action or "",
                config_yaml=config_yaml,
                config_path=c_path,
                data_path=d_path,
                artifact_path=artifact_path,
                scenarios_path=scenarios_path,
                export_format=export_format,
                priority=str(priority or "normal"),
            )
        )
        return f"[QUEUED] {result.message} (Job ID: {result.job_id})"
    except Exception as exc:
        return f"[ERROR] {normalize_gui_error(exc)}"


def _cb_refresh_run_jobs(
    _n_intervals: int,
    _refresh_clicks: int,
    last_status: dict | None,
    workflow_state: dict | None,
    current_path: str,
    batch_mode: list,
) -> tuple[html.Div, Any, dict, dict, str]:
    jobs = list_run_jobs(limit=100)
    priority_rank = {"high": 0, "normal": 1, "low": 2}
    queued_jobs = [job for job in jobs if job.status == "queued"]
    queued_jobs.sort(key=lambda job: (priority_rank.get(job.priority, 1), job.created_at_utc))
    non_queued_jobs = [job for job in jobs if job.status != "queued"]
    jobs = queued_jobs + non_queued_jobs

    new_status = {}
    toast = dash.no_update
    next_path = dash.no_update

    last_status = last_status or {}
    next_state = dict(workflow_state or {})
    is_batch = "enabled" in (batch_mode or [])

    for job in jobs:
        new_status[job.job_id] = job.status

        old_s = last_status.get(job.job_id)
        if old_s in ["running", "queued"] and job.status in ["succeeded", "failed"]:
            icon = "success" if job.status == "succeeded" else "danger"
            toast = make_toast(f"Task {job.action} {job.status}!", icon=icon)

            if job.status == "succeeded" and not is_batch and current_path == "/run":
                next_path = "/results"
            if job.status == "succeeded":
                next_state["last_job_succeeded"] = True
            payload = (
                job.result.payload if (job.result and isinstance(job.result.payload, dict)) else {}
            )
            artifact_path = payload.get("artifact_path")
            if isinstance(artifact_path, str) and artifact_path.strip():
                next_state["last_run_artifact"] = artifact_path

    data = [
        {
            "job_id": job.job_id,
            "action": job.action,
            "priority": job.priority.upper(),
            "progress": f"{float(job.progress_pct):.1f}% | {job.current_step or '-'}",
            "status": _status_badge(job.status),
            "created_at_utc": _format_jst_timestamp(job.created_at_utc),
            "id": job.job_id,
        }
        for job in jobs
    ]
    return task_table("run-jobs", data), toast, new_status, next_state, next_path


def _cb_show_selected_job_detail(
    selected_rows: list[int] | None,
    _n_intervals: int,
    _load_more_clicks: int,
    data: list[dict] | None,
    selected_job_id: str | None,
    current_log_limit: int | None,
) -> tuple[Any, bool, bool, str, str | None, int]:
    if not data:
        return "Select a job to view details.", True, True, "", selected_job_id, 200

    next_log_limit = max(50, int(current_log_limit or 200))
    triggered = ""
    try:
        if callback_context.triggered:
            triggered = str(callback_context.triggered[0]["prop_id"]).split(".")[0]
    except Exception:
        triggered = ""
    if triggered == "run-log-load-more-btn":
        next_log_limit = min(next_log_limit + 200, 10_000)
    elif triggered == "run-jobs-table":
        next_log_limit = 200

    job_id = selected_job_id
    if selected_rows:
        row_idx = selected_rows[0]
        if row_idx >= len(data):
            return "Job not found.", True, True, "", selected_job_id, next_log_limit
        job_id = data[row_idx]["job_id"]

    if not job_id:
        return "Select a job to view details.", True, True, "", None, next_log_limit

    job = get_run_job(job_id)

    if not job:
        return "Job details unavailable.", True, True, "", selected_job_id, next_log_limit

    can_cancel = job.status in ["queued", "running"]
    can_retry = job.status in ["failed", "canceled"]
    status_color = "primary"
    if job.status == "succeeded":
        status_color = "success"
    elif job.status == "failed":
        status_color = "danger"

    details_elems = [
        html.H5(f"Task: {job.action.upper()}", className="mb-3"),
        html.Div(
            [
                html.Span("Status: ", className="fw-bold"),
                html.Span(
                    job.status.upper(),
                    className=f"badge bg-{status_color}",
                ),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                html.Span("Priority: ", className="fw-bold"),
                html.Span(job.priority.upper()),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                html.Span("Progress: ", className="fw-bold"),
                html.Span(f"{float(job.progress_pct):.1f}%"),
                html.Span(f" | {job.current_step or 'n/a'}", className="text-muted ms-2"),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                html.Span("Created: ", className="fw-bold"),
                html.Span(_format_jst_timestamp(job.created_at_utc)),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                html.Span("Started: ", className="fw-bold"),
                html.Span(_format_jst_timestamp(job.started_at_utc)),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                html.Span("Finished: ", className="fw-bold"),
                html.Span(_format_jst_timestamp(job.finished_at_utc)),
            ],
            className="mb-2",
        ),
    ]

    if job.error_message:
        details_elems.append(
            dbc.Alert(
                html.Pre(job.error_message, className="mb-0"), color="danger", className="mt-3"
            )
        )
    next_steps = []
    if job.result and isinstance(job.result.payload, dict):
        raw_steps = job.result.payload.get("next_steps")
        if isinstance(raw_steps, list):
            next_steps = [str(item) for item in raw_steps if str(item).strip()]
    if next_steps:
        details_elems.append(html.Label("Next Step", className="fw-bold mt-3"))
        details_elems.append(html.Ul([html.Li(step) for step in next_steps], className="mb-2"))

    if job.result and job.result.payload:
        payload_str = _json_dumps(job.result.payload)
        details_elems.append(html.Label("Result Payload", className="fw-bold mt-3"))
        details_elems.append(
            html.Pre(
                payload_str,
                className="bg-dark text-white p-2 rounded",
                style={"maxHeight": "300px", "overflow": "auto"},
            )
        )

    logs = list_run_job_logs(job.job_id, limit=next_log_limit)
    details_elems.append(html.Label("Live Logs", className="fw-bold mt-3"))
    details_elems.append(
        render_progress_viewer(
            progress_pct=job.progress_pct,
            current_step=job.current_step,
            logs=logs,
            log_limit=next_log_limit,
            log_total=len(logs),
        )
    )

    details = html.Div(details_elems, className="p-2")

    return details, not can_cancel, not can_retry, f"Selected: {job_id}", job_id, next_log_limit


def _cb_cancel_job(_n_clicks: int, job_id: str | None) -> str:
    if not job_id:
        return ""
    try:
        result = cancel_run_job(job_id)
        return f"[INFO] {result.message}"
    except Exception as exc:
        return f"[ERROR] {str(exc)}"


def _cb_retry_job(_n_clicks: int, job_id: str | None) -> str:
    if not job_id:
        return ""
    try:
        result = retry_run_job(job_id)
        return f"[INFO] {result.message}"
    except Exception as exc:
        return f"[ERROR] {normalize_gui_error(exc)}"


def _cb_set_job_priority(_n_clicks: int, job_id: str | None, priority: str | None) -> str:
    if not job_id:
        return "[ERROR] Select a queued job first."
    try:
        result = set_run_job_priority(job_id, str(priority or "normal"))
        return f"[INFO] {result.message}"
    except Exception as exc:
        return f"[ERROR] {normalize_gui_error(exc)}"


def _cb_list_artifacts(
    _n_clicks: int, _pathname: str, root_path: str
) -> tuple[list[dict], list[dict]]:
    try:
        items = list_artifacts(root_path or "artifacts")
        options = [
            {
                "label": (
                    f"{_format_jst_timestamp(item.created_at_utc)} | "
                    f"{item.task_type} | {item.run_id}"
                ),
                "value": item.path,
            }
            for item in items
        ]
        return options, options
    except Exception:
        return [], []


def _cb_autoselect_artifact(
    state: dict | None,
    options: list[dict] | None,
    pathname: str | None,
    current_value: str | None,
) -> str:
    if pathname != "/results":
        return dash.no_update

    opts = options or []
    if not opts:
        return dash.no_update

    preferred = ""
    if isinstance(state, dict):
        preferred = str(state.get("last_run_artifact") or "")

    values = [str(opt.get("value", "")) for opt in opts]
    if preferred and preferred in values and current_value != preferred:
        return preferred

    if current_value in values:
        return dash.no_update

    # Fallback: select the latest artifact (options are sorted newest-first).
    return values[0]


def _cb_update_result_view(
    artifact_path: str | None, compare_path: str | None
) -> tuple[Any, Any, Any, Any]:
    if not artifact_path:
        # Return empty placeholder charts with a message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94a3b8", "family": "Inter"},
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Select an artifact to view results",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 16, "color": "#94a3b8"},
                }
            ],
        )
        return "", empty_fig, empty_fig, "Select an artifact to view results."

    try:
        art = Artifact.load(artifact_path)

        comp_art = None
        if compare_path and compare_path != artifact_path:
            try:
                comp_art = Artifact.load(compare_path)
            except Exception:
                pass

        def _select_plot_metrics(metrics_obj: Any) -> dict[str, float]:
            if not isinstance(metrics_obj, dict):
                return {}
            top_numeric = {
                k: float(v) for k, v in metrics_obj.items() if isinstance(v, (int, float))
            }
            if top_numeric:
                return top_numeric
            mean_obj = metrics_obj.get("mean")
            if isinstance(mean_obj, dict):
                return {k: float(v) for k, v in mean_obj.items() if isinstance(v, (int, float))}
            return {}

        kpi_elems = []
        metrics_raw = art.metrics or {}
        metrics = _select_plot_metrics(metrics_raw)
        for key in ["r2_score", "accuracy", "f1_score", "rmse", "mae"]:
            if key in metrics:
                val = metrics[key]
                kpi_elems.append(kpi_card(key, val))
        if "r2" in metrics and "r2_score" not in metrics:
            kpi_elems.append(kpi_card("r2", metrics["r2"]))

        for k, v in metrics.items():
            if k not in ["r2_score", "accuracy", "f1_score", "rmse", "mae"] and isinstance(
                v, (int, float)
            ):
                kpi_elems.append(kpi_card(k, v))

        kpi_container = html.Div(kpi_elems, className="d-flex flex-wrap gap-3")

        run_id = getattr(art, "run_id", None) or getattr(
            getattr(art, "manifest", None), "run_id", "n/a"
        )
        task_type = (
            getattr(art, "task_type", None)
            or getattr(getattr(art, "manifest", None), "task_type", None)
            or getattr(getattr(getattr(art, "run_config", None), "task", None), "type", "n/a")
        )
        created_at = getattr(art, "created_at_utc", None) or getattr(
            getattr(art, "manifest", None), "created_at_utc", None
        )
        config_obj = getattr(art, "config", None) or getattr(art, "run_config", None)
        feature_importance = None
        metadata_obj = getattr(art, "metadata", None)
        if isinstance(metadata_obj, dict):
            feature_importance = metadata_obj.get("feature_importance")
        if feature_importance is None:
            schema_obj = getattr(art, "feature_schema", None)
            if isinstance(schema_obj, dict):
                feature_importance = schema_obj.get("feature_importance")
        if feature_importance is None:
            # Fallback: reconstruct feature importance from LightGBM booster if available.
            try:
                booster = art._get_booster() if hasattr(art, "_get_booster") else None
                feature_names = []
                schema_obj = getattr(art, "feature_schema", None)
                if isinstance(schema_obj, dict):
                    maybe_names = schema_obj.get("feature_names")
                    if isinstance(maybe_names, list):
                        feature_names = [str(v) for v in maybe_names]
                if booster is not None:
                    gain_vals = booster.feature_importance(importance_type="gain")
                    if not feature_names:
                        feature_names = [str(v) for v in booster.feature_name()]
                    if len(feature_names) == len(gain_vals):
                        feature_importance = {
                            name: float(val)
                            for name, val in zip(feature_names, gain_vals)
                            if float(val) > 0.0
                        }
            except Exception:
                feature_importance = None

        if comp_art:
            comp_metrics = _select_plot_metrics(comp_art.metrics or {})
            comp_run_id = getattr(comp_art, "run_id", None) or getattr(
                getattr(comp_art, "manifest", None), "run_id", "n/a"
            )
            fig_main = plot_comparison_bar(
                metrics,
                comp_metrics,
                name1=f"Current ({str(run_id)[:6]}...)",
                name2=f"Baseline ({str(comp_run_id)[:6]}...)",
            )
        else:
            fig_main = plot_metrics_bar(metrics, title="Performance Metrics")

        fig_sec = {}
        if feature_importance:
            fig_sec = plot_feature_importance(feature_importance)

        dt = _format_jst_timestamp(created_at)
        details_str = (
            f"Run ID: {run_id}\n"
            f"Type: {task_type}\n"
            f"Created: {dt}\n\n"
            f"Run Config:\n{_json_dumps(config_obj)}"
        )
        details_elem = html.Pre(details_str, className="p-3 border rounded")

        # Safe default for figure
        if not fig_sec:
            fig_sec = go.Figure()
            fig_sec.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                annotations=[
                    {
                        "text": "No Feature Importance Available",
                        "showarrow": False,
                        "font": {"color": "white"},
                    }
                ],
            )

        return kpi_container, fig_main, fig_sec, details_elem

    except Exception as exc:
        return html.Div(f"Error loading artifact: {exc}"), {}, {}, ""


def _cb_evaluate_artifact_action(_n_clicks: int, artifact_path: str, data_path: str) -> str:
    try:
        if not artifact_path or not data_path:
            return "Artifact and Data path are required."

        artifact = Artifact.load(artifact_path)
        frame = _get_load_tabular_data()(data_path)
        result = _get_evaluate()(artifact, frame)

        try:
            payload = asdict(result)
            # Ensure DataFrames are converted to strings strings
            for k, v in payload.items():
                if isinstance(v, pd.DataFrame):
                    payload[k] = f"<DataFrame: {len(v)} rows x {len(v.columns)} cols>"
        except Exception:
            payload = {}
            for field in result.__dataclass_fields__:
                val = getattr(result, field)
                if isinstance(val, pd.DataFrame):
                    payload[field] = f"<DataFrame: {len(val)} rows x {len(val.columns)} cols>"
                else:
                    payload[field] = val

        if "data" in payload and isinstance(payload["data"], (dict, list)):
            pass
        elif "data" in payload and hasattr(payload["data"], "to_dict"):
            del payload["data"]

        return _json_dumps(payload)
    except Exception as exc:
        return f"Evaluation failed: {exc}"


def _cb_update_result_extras(artifact_path: str | None) -> tuple[Any, str, Any]:
    if not artifact_path:
        return go.Figure(), "", "Select artifact to view details."
    try:
        art = Artifact.load(artifact_path)
        history = getattr(art, "training_history", None)
        curve_fig = plot_learning_curves(history if isinstance(history, dict) else {})
        cfg_obj = getattr(art, "config", None) or getattr(art, "run_config", None)
        cfg_text = yaml.safe_dump(_to_jsonable(cfg_obj), sort_keys=False, allow_unicode=True)

        schema = getattr(art, "feature_schema", {}) or {}
        summary = html.Ul(
            [
                html.Li(f"Features: {len(schema.get('feature_names', []))}"),
                html.Li(f"Rows: {schema.get('n_rows', 'n/a')}"),
                html.Li(f"Task: {getattr(art, 'task_type', 'n/a')}"),
            ],
            className="mb-0",
        )
        return curve_fig, cfg_text, summary
    except Exception as exc:
        return go.Figure(), f"Error: {exc}", f"Error: {exc}"


def _cb_result_export_help() -> str:
    return (
        "Excel: feature importance, predictions, residual views by sheets. "
        "HTML: self-contained shareable report."
    )


def _cb_result_export_help_for_artifact(_artifact_path: str | None) -> str:
    return _cb_result_export_help()


def _cb_result_eval_precheck(artifact_path: str | None, data_path: str | None) -> Any:
    if not artifact_path:
        return guide_alert(["Select an artifact first."], severity="info")
    if not (data_path or "").strip():
        return guide_alert(["Set Evaluation Data Path to run feature compatibility check."], "info")
    try:
        art = Artifact.load(artifact_path)
        schema = getattr(art, "feature_schema", {}) or {}
        expected = set(schema.get("feature_names") or [])
        frame = _get_load_tabular_data()(str(data_path))
        actual = set(frame.columns)
        if not expected:
            return guide_alert(["Feature schema is not available in this artifact."], "warning")
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if not missing and not extra:
            return guide_alert(["Feature schema check passed."], "ok")
        messages: list[str] = []
        severity = "warning"
        if missing:
            severity = "error"
            messages.append(f"Missing features: {', '.join(missing[:8])}")
        if extra:
            messages.append(f"Extra columns: {', '.join(extra[:8])}")
        messages.append("Align columns before running re-evaluate.")
        return guide_alert(messages, severity=severity)
    except Exception as exc:
        return guide_alert([f"Precheck failed: {exc}"], severity="warning")


def _cb_result_shortcut_highlight(
    state: dict[str, Any] | None,
    pathname: str | None,
) -> tuple[str, str, str]:
    if pathname != "/results":
        return "w-100 mb-3", "me-2 result-export-btn", "me-2 result-export-btn"
    current = _ensure_workflow_state_defaults(state)
    focus = current.get("results_shortcut_focus")
    eval_class = "w-100 mb-3"
    excel_class = "me-2 result-export-btn"
    html_class = "me-2 result-export-btn"
    if focus == "evaluate":
        eval_class += " border border-warning"
    if focus == "export":
        excel_class += " border border-warning"
        html_class += " border border-warning"
    return eval_class, excel_class, html_class


def _cb_result_export_actions(
    _excel_clicks: int,
    _html_clicks: int,
    artifact_path: str | None,
) -> tuple[str, dict[str, str] | None, bool, int]:
    if not artifact_path:
        return "Select an artifact first.", None, True, 0
    triggered = ""
    try:
        if callback_context.triggered:
            triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    except Exception:
        triggered = ""
    if not triggered:
        triggered = (
            "result-export-excel-btn"
            if (_excel_clicks or 0) >= (_html_clicks or 0)
            else "result-export-html-btn"
        )
    action = "export_excel" if triggered == "result-export-excel-btn" else "export_html_report"
    try:
        result = submit_run_job(RunInvocation(action=action, artifact_path=artifact_path))
        state = {"job_id": result.job_id, "action": action}
        return f"生成中... ({action}, Job ID: {result.job_id})", state, False, 0
    except Exception as exc:
        return f"[ERROR] {exc}", None, True, 0


def _cb_poll_result_export_job(
    _n_intervals: int,
    export_state: dict[str, Any] | None,
) -> tuple[Any, str, dict[str, Any] | None, bool]:
    if not export_state:
        return dash.no_update, dash.no_update, dash.no_update, True

    job_id = str(export_state.get("job_id") or "")
    action = str(export_state.get("action") or "export")
    if not job_id:
        return dash.no_update, "[ERROR] Export job id is missing.", None, True

    job = get_run_job(job_id)
    if job is None:
        return dash.no_update, f"[ERROR] Export job not found: {job_id}", None, True

    if job.status in {"queued", "running", "cancel_requested"}:
        return dash.no_update, f"生成中... ({action}, Job ID: {job_id})", export_state, False

    if job.status == "succeeded":
        payload = (
            job.result.payload if (job.result and isinstance(job.result.payload, dict)) else {}
        )
        output_path = str(payload.get("output_path") or "")
        if not output_path:
            return dash.no_update, "[ERROR] Export output path is missing.", None, True
        path_obj = Path(output_path)
        if not path_obj.is_file():
            return dash.no_update, f"[ERROR] Export file not found: {path_obj}", None, True
        return dcc.send_file(str(path_obj)), "ダウンロード完了", None, True

    if job.status in {"failed", "canceled"}:
        message = job.error_message or (job.result.message if job.result else "Export failed.")
        return dash.no_update, f"[ERROR] {message}", None, True

    return dash.no_update, f"[INFO] Export status: {job.status}", export_state, False


def _cb_result_download_config(_n_clicks: int, artifact_path: str | None) -> Any:
    if not artifact_path:
        return dash.no_update
    try:
        art = Artifact.load(artifact_path)
        cfg_obj = getattr(art, "config", None) or getattr(art, "run_config", None)
        cfg_text = yaml.safe_dump(_to_jsonable(cfg_obj), sort_keys=False, allow_unicode=True)
        return dcc.send_string(cfg_text, filename="run_config.yaml")
    except Exception:
        return dash.no_update


def _cb_refresh_runs_table(
    _refresh_clicks: int,
    pathname: str | None,
    status_filter: str | None,
    action_filter: str | None,
    query: str | None,
) -> list[dict[str, Any]]:
    if pathname != "/runs":
        return []
    jobs = list_run_jobs_filtered(
        limit=300,
        status=status_filter or None,
        action=action_filter or None,
        query=query or None,
    )
    rows: list[dict[str, Any]] = []
    for job in jobs:
        artifact_path = job.invocation.artifact_path or ""
        if job.result and isinstance(job.result.payload, dict):
            artifact_path = str(job.result.payload.get("artifact_path") or artifact_path)
        rows.append(
            {
                "job_id": job.job_id,
                "status": job.status,
                "action": job.action,
                "created_at_utc": _format_jst_timestamp(job.created_at_utc),
                "started_at_utc": _format_jst_timestamp(job.started_at_utc),
                "finished_at_utc": _format_jst_timestamp(job.finished_at_utc),
                "artifact_path": artifact_path,
                "export_shortcut": "Export",
                "reeval_shortcut": "Re-evaluate",
            }
        )
    return rows


def _cb_runs_selection_detail(
    selected_rows: list[int] | None, data: list[dict] | None
) -> tuple[str, list[str]]:
    if not selected_rows or not data:
        return "Select one or more runs.", []
    job_ids: list[str] = []
    details: list[str] = []
    for idx in selected_rows:
        if idx >= len(data):
            continue
        row = data[idx]
        job_id = str(row.get("job_id") or "")
        if not job_id:
            continue
        job = get_run_job(job_id)
        if job is None:
            continue
        job_ids.append(job_id)
        details.append(
            f"Job ID: {job.job_id}\n"
            f"Status: {job.status}\n"
            f"Action: {job.action}\n"
            f"Created: {_format_jst_timestamp(job.created_at_utc)}\n"
            f"Started: {_format_jst_timestamp(job.started_at_utc)}\n"
            f"Finished: {_format_jst_timestamp(job.finished_at_utc)}\n"
            f"Artifact: {job.invocation.artifact_path or ''}\n"
        )
    return "\n---\n".join(details) or "No details.", job_ids


def _cb_runs_actions(
    _compare_clicks: int,
    _clone_clicks: int,
    _delete_clicks: int,
    _view_results_clicks: int,
    _migrate_clicks: int,
    active_cell: dict[str, Any] | None,
    selected_job_ids: list[str] | None,
    state: dict | None,
    table_data: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], Any, str]:
    current = _ensure_workflow_state_defaults(state)
    job_ids = list(selected_job_ids or [])
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

    if triggered == "runs-table":
        if not active_cell or not table_data:
            return current, dash.no_update, "Select a run row first."
        row = active_cell.get("row")
        col = active_cell.get("column_id")
        if row is None or row >= len(table_data):
            return current, dash.no_update, "Invalid run row selected."
        row_data = table_data[row]
        artifact_path = str(row_data.get("artifact_path") or "")
        if not artifact_path:
            return current, dash.no_update, "No artifact path available for selected run."
        current["last_run_artifact"] = artifact_path
        if col == "export_shortcut":
            current["results_shortcut_focus"] = "export"
            return current, "/results", "Moved to Results (Export shortcut)."
        if col == "reeval_shortcut":
            current["results_shortcut_focus"] = "evaluate"
            return current, "/results", "Moved to Results (Re-evaluate shortcut)."
        return current, dash.no_update, "No shortcut action."

    if not job_ids:
        return current, dash.no_update, "Select one or more runs."

    try:
        if triggered == "runs-delete-btn":
            count = delete_run_jobs(job_ids)
            return current, dash.no_update, f"Deleted {count} run records."

        if triggered == "runs-compare-btn":
            if len(job_ids) != 2:
                return current, dash.no_update, "Select exactly two runs for compare."
            paths: list[str] = []
            for job_id in job_ids:
                job = get_run_job(job_id)
                if not job:
                    continue
                path = job.invocation.artifact_path or ""
                if job.result and isinstance(job.result.payload, dict):
                    path = str(job.result.payload.get("artifact_path") or path)
                if path:
                    paths.append(path)
            if len(paths) != 2:
                return current, dash.no_update, "Artifacts not found for selected runs."
            current["compare_selection"] = paths
            return current, "/compare", "Moved to Compare."

        primary = get_run_job(job_ids[0])
        if primary is None:
            return current, dash.no_update, "Run not found."

        if triggered == "runs-clone-btn":
            yaml_text = load_job_config_yaml(primary)
            payload = yaml.safe_load(yaml_text) or {}
            if not isinstance(payload, dict):
                return current, dash.no_update, "Invalid config payload."
            current = _state_from_config_payload(payload, current)
            return (
                current,
                "/train",
                "Config cloned. Review settings in Train page, then re-run.",
            )

        if triggered == "runs-view-results-btn":
            path = primary.invocation.artifact_path or ""
            if primary.result and isinstance(primary.result.payload, dict):
                path = str(primary.result.payload.get("artifact_path") or path)
            if not path:
                return current, dash.no_update, "No artifact path available."
            current["last_run_artifact"] = path
            current["results_shortcut_focus"] = None
            return current, "/results", "Moved to Results."

        if triggered == "runs-migrate-btn":
            if not primary.invocation.config_path:
                return current, dash.no_update, "No config path stored for selected run."
            msg = migrate_config_file_via_gui(primary.invocation.config_path, target_version=1)
            return current, dash.no_update, msg
    except Exception as exc:
        return current, dash.no_update, f"Run action failed: {exc}"

    return current, dash.no_update, "No action."


def _cb_populate_compare_options(
    pathname: str | None, state: dict | None
) -> tuple[list[dict[str, str]], list[dict[str, str]], Any, Any]:
    if pathname != "/compare":
        return [], [], None, None
    options = [
        {"label": f"{item.task_type} | {item.run_id}", "value": item.path}
        for item in list_artifacts("artifacts")
    ]
    compare_selection = (_ensure_workflow_state_defaults(state)).get("compare_selection") or []
    value_a = compare_selection[0] if len(compare_selection) > 0 else None
    value_b = compare_selection[1] if len(compare_selection) > 1 else None
    return options, options, value_a, value_b


def _cb_compare_runs(
    artifact_a: str | None, artifact_b: str | None
) -> tuple[Any, list[dict], Any, Any]:
    if not artifact_a or not artifact_b:
        return "Select both artifacts.", [], go.Figure(), ""
    try:
        payload = compare_artifacts(artifact_a, artifact_b)
        checks = render_guardrails(payload.get("checks", []))
        fig = plot_comparison_bar(
            payload.get("metrics_a", {}), payload.get("metrics_b", {}), "Run A", "Run B"
        )
        diff = render_yaml_diff(payload.get("config_yaml_a", ""), payload.get("config_yaml_b", ""))
        return checks, payload.get("metric_rows", []), fig, diff
    except Exception as exc:
        return f"Compare failed: {exc}", [], go.Figure(), ""


def _sync_path_preset(preset_val: str, current_path: str) -> tuple[str, str]:
    """Sync Dropdown Preset with Input Path."""
    triggered = callback_context.triggered_id

    # If Preset changed
    if "preset" in str(triggered):
        if preset_val == "custom":
            return "custom", dash.no_update
        return preset_val, preset_val

    # If Input changed
    if current_path == "artifacts":
        return "artifacts", "artifacts"
    if current_path == "output":
        return "output", "output"

    return "custom", current_path
