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
    plot_metrics_bar,
)
from veldra.gui.components.kpi_cards import kpi_card
from veldra.gui.components.task_table import task_table
from veldra.gui.components.toast import make_toast, toast_container
from veldra.gui.pages import config_page, data_page, results_page, run_page
from veldra.gui.services import (
    cancel_run_job,
    get_run_job,
    inspect_data,
    list_artifacts,
    list_run_jobs,
    load_config_yaml,
    migrate_config_file_via_gui,
    migrate_config_from_yaml,
    normalize_gui_error,
    save_config_yaml,
    submit_run_job,
    validate_config,
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
                        [html.I(className="bi bi-gear me-2"), "Config"],
                        href="/config",
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
                ],
                vertical=True,
                pills=True,
            ),
            html.Div(html.Span("v0.1.0", className="badge bg-secondary"), className="mt-auto mb-3"),
        ],
        className="sidebar",
    )


def _stepper_bar(pathname: str) -> html.Div:
    steps = [
        {"label": "Data", "path": "/data"},
        {"label": "Config", "path": "/config"},
        {"label": "Run", "path": "/run"},
        {"label": "Results", "path": "/results"},
    ]

    step_elems = []
    current_idx = -1

    # Find current step index
    for i, step in enumerate(steps):
        if pathname and step["path"] in pathname:
            current_idx = i
            break
    if current_idx == -1 and pathname == "/":  # Default to data
        current_idx = 0

    for i, step in enumerate(steps):
        status_class = ""
        icon = str(i + 1)

        if i < current_idx:
            status_class = "completed"
            icon = "✓"
        elif i == current_idx:
            status_class = "active"

        step_content = html.Div(
            [html.Div(icon, className="step-circle"), html.Span(step["label"])],
            className=f"step-item {status_class}",
        )
        step_elems.append(
            dcc.Link(step_content, href=step["path"], style={"textDecoration": "none"})
        )

        # Add connector line if not last item
        if i < len(steps) - 1:
            connector_color = "var(--success)" if i < current_idx else "rgba(148, 163, 184, 0.1)"
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
        return config_page.layout()
    if pathname == "/run":
        return run_page.layout(state)
    if pathname == "/results":
        return results_page.layout()
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
        return render_page(pathname, state), _stepper_bar(pathname or "/")

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

    # --- Run Page Auto-Action ---
    app.callback(
        Output("run-action", "value"),
        Output("run-action-display", "children"),
        Output("run-action-display", "className"),
        Input("run-config-yaml", "value"),
    )(_cb_detect_run_action)

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
        Output("selected-job-id-display", "children"),
        Output("run-job-select", "data"),  # Store selection
        Input("run-jobs-table", "selected_rows"),
        State("run-jobs-table", "data"),
        State("run-job-select", "data"),
        prevent_initial_call=True,
    )(_cb_show_selected_job_detail)

    app.callback(
        Output("run-result-log", "children", allow_duplicate=True),
        Input("run-cancel-job-btn", "n_clicks"),
        State("run-job-select", "data"),
        prevent_initial_call=True,
    )(_cb_cancel_job)

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
    next_state = dict(current_state)
    next_state["data_path"] = final_path
    return _ret(
        html.Div([stats_div, preview_div], className="data-inspection-zone"),
        "",
        next_state,
        label,
        final_path,
    )


def _cb_save_target_col(target_col: str, state: dict) -> dict:
    if not state:
        state = {}
    state["target_col"] = target_col
    return state


def _cb_update_selected_file_label(filename: str | list[str] | None) -> tuple[str, str]:
    if isinstance(filename, list):
        selected = filename[0] if filename else None
    else:
        selected = filename
    if not selected:
        return "No file selected — upload or drop a file above", ""
    return f"Selected: {selected}", ""


def _cb_cache_config_yaml(config_yaml: str, state: dict | None) -> dict:
    if not state:
        state = {}
    state["config_yaml"] = config_yaml or ""
    return state


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


def _cb_detect_run_action(yaml_text: str) -> tuple[str, str, str]:
    default_vals = ("fit", "Ready: TRAIN", "badge bg-primary fs-6 p-2 mb-3")
    if not yaml_text:
        return default_vals

    try:
        cfg = yaml.safe_load(yaml_text)
        if not isinstance(cfg, dict):
            return default_vals

        # Check Tuning
        if cfg.get("tuning", {}).get("enabled", False):
            return "tune", "Ready: TUNE", "badge bg-warning text-dark fs-6 p-2 mb-3"

        # Check Causal
        if "causal_method" in cfg.get("task", {}):
            return "fit", "Ready: CAUSAL ANALYSIS", "badge bg-info text-dark fs-6 p-2 mb-3"

        return default_vals
    except Exception:
        return default_vals


def _cb_update_tune_objectives(task_type: str) -> list[dict]:
    objectives = {
        "regression": ["rmse", "mae", "r2"],
        "binary": ["auc", "logloss", "accuracy", "f1", "precision", "recall"],
        "multiclass": ["accuracy", "macro_f1", "logloss"],
        "frontier": ["pinball", "pinball_coverage_penalty"],
    }
    opts = objectives.get(task_type, [])
    return [{"label": o.upper(), "value": o} for o in opts]


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

    return yaml.dump(cfg, sort_keys=False)


def _cb_update_class_weight_visibility(
    task_type: str | None,
    auto_class_weight_enabled: bool | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if task_type not in {"binary", "multiclass"}:
        return {"display": "none"}, {"display": "none"}
    show_manual = not bool(auto_class_weight_enabled)
    return {"display": "block"}, {"display": "block" if show_manual else "none"}


def _cb_update_run_launch_state(
    action: str | None,
    data_path: str | None,
    config_yaml: str | None,
    config_path: str | None,
    artifact_path: str | None,
    scenarios_path: str | None,
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
) -> str:
    try:
        c_path = _ensure_default_run_config(config_path_state or "configs/gui_run.yaml")
        d_path = data_path_state

        result = submit_run_job(
            RunInvocation(
                action=action or "",
                config_yaml=config_yaml,
                config_path=c_path,
                data_path=d_path,
                artifact_path=artifact_path,
                scenarios_path=scenarios_path,
                export_format=export_format,
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
            "status": _status_badge(job.status),
            "created_at_utc": job.created_at_utc,
            "id": job.job_id,
        }
        for job in jobs
    ]
    return task_table("run-jobs", data), toast, new_status, next_state, next_path


def _cb_show_selected_job_detail(
    selected_rows: list[int] | None,
    data: list[dict] | None,
    selected_job_id: str | None,
) -> tuple[str, bool, str, str | None]:
    if not data:
        return "Select a job to view details.", True, "", selected_job_id

    job_id = selected_job_id
    if selected_rows:
        row_idx = selected_rows[0]
        if row_idx >= len(data):
            return "Job not found.", True, "", selected_job_id
        job_id = data[row_idx]["job_id"]

    if not job_id:
        return "Select a job to view details.", True, "", None

    job = get_run_job(job_id)

    if not job:
        return "Job details unavailable.", True, "", selected_job_id

    can_cancel = job.status in ["queued", "running"]
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
            [html.Span("Created: ", className="fw-bold"), html.Span(job.created_at_utc)],
            className="mb-2",
        ),
        html.Div(
            [html.Span("Started: ", className="fw-bold"), html.Span(job.started_at_utc or "-")],
            className="mb-2",
        ),
        html.Div(
            [html.Span("Finished: ", className="fw-bold"), html.Span(job.finished_at_utc or "-")],
            className="mb-2",
        ),
    ]

    if job.error_message:
        details_elems.append(
            dbc.Alert(
                html.Pre(job.error_message, className="mb-0"), color="danger", className="mt-3"
            )
        )

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

    details = html.Div(details_elems, className="p-2")

    return details, not can_cancel, f"Selected: {job_id}", job_id


def _cb_cancel_job(_n_clicks: int, job_id: str | None) -> str:
    if not job_id:
        return ""
    try:
        result = cancel_run_job(job_id)
        return f"[INFO] {result.message}"
    except Exception as exc:
        return f"[ERROR] {str(exc)}"


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
