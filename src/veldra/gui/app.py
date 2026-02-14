"""Dash app factory for Veldra GUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import yaml
from dataclasses import asdict, is_dataclass
from typing import Any
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html
import plotly.graph_objs as go

from veldra.api.artifact import Artifact
from veldra.api.runner import evaluate
from veldra.data import load_tabular_data
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
    run_action,
    save_config_yaml,
    submit_run_job,
    validate_config,
)
from veldra.gui.types import RunInvocation
from veldra.gui.components.charts import (
    plot_feature_importance, 
    plot_actual_vs_predicted,
    plot_metrics_bar,
    plot_comparison_bar
)
from veldra.gui.components.task_table import task_table
from veldra.gui.components.kpi_cards import kpi_card


def _sidebar() -> html.Div:
    return html.Div(
        [
            html.H3("Veldra", className="text-white mb-4 text-center fw-bold", style={"letterSpacing": "2px"}),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [html.I(className="bi bi-database me-2"), "Data"], 
                        href="/data", active="exact", className="nav-link"
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-gear me-2"), "Config"], 
                        href="/config", active="exact", className="nav-link"
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-play-circle me-2"), "Run"], 
                        href="/run", active="exact", className="nav-link"
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-graph-up me-2"), "Results"], 
                        href="/results", active="exact", className="nav-link"
                    ),
                ],
                vertical=True,
                pills=True,
            ),
            html.Div(
                html.Span("v0.1.0", className="badge bg-secondary"),
                className="mt-auto mb-3"
            )
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
    if current_idx == -1 and pathname == "/": # Default to data
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
            [
                html.Div(icon, className="step-circle"),
                html.Span(step["label"])
            ],
            className=f"step-item {status_class}"
        )
        step_elems.append(
            dcc.Link(step_content, href=step["path"], style={"textDecoration": "none"})
        )
        
        # Add connector line if not last item
        if i < len(steps) - 1:
            step_elems.append(
                html.Div(
                    style={
                        "flexGrow": "1", 
                        "height": "2px", 
                        "backgroundColor": "rgba(148, 163, 184, 0.1)",
                        "margin": "0 10px"
                    }
                )
            )

    return html.Div(step_elems, className="stepper-container")


from veldra.gui.components.toast import toast_container, make_toast

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
                        style={"flex": "1", "minHeight": "100vh"},
                    ),
                ],
                className="g-0",
                style={"minHeight": "100vh", "overflow": "visible"}
            ),
        ]
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


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)



def _status_badge(status: str) -> str:
    return status.upper()


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP, 
            dbc.icons.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap"
        ],
        suppress_callback_exceptions=True,
        title="Veldra GUI",
    )
    app.layout = _main_layout()
    poll_ms = max(200, int(os.getenv("VELDRA_GUI_POLL_MS", "2000")))

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
        Input("data-inspect-btn", "n_clicks"),
        State("data-upload-drag", "contents"),
        State("data-upload-drag", "filename"),
        prevent_initial_call=True,
    )(_cb_inspect_data)

    app.callback(
        Output("workflow-state", "data", allow_duplicate=True),
        Input("data-target-col", "value"),
        State("workflow-state", "data"),
        prevent_initial_call=True,
    )(_cb_save_target_col)

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
    )(_cb_update_tune_visibility) # Reuse same logic (block/none)

    app.callback(
        Output("cfg-container-id-cols", "style"),
        Input("cfg-split-type", "value"),
    )(lambda st: {"display": "block"} if st == "group" else {"display": "none"})

    # --- Run Page Auto-Action ---
    app.callback(
        Output("run-action", "value"),
        Output("run-action-display", "children"),
        Output("run-action-display", "className"),
        Input("run-config-yaml", "value"),
    )(_cb_detect_run_action)




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
        Output("url", "pathname"), # Auto-navigation
        Input("run-jobs-interval", "n_intervals"),
        Input("run-refresh-jobs-btn", "n_clicks"),
        State("last-job-status", "data"),
        State("url", "pathname"),
        State("run-batch-mode-toggle", "value"),
        prevent_initial_call=True
    )(_cb_refresh_run_jobs)

    app.callback(
        Output("run-job-detail", "children"),
        Output("run-cancel-job-btn", "disabled"),
        Output("selected-job-id-display", "children"),
        Output("run-job-select", "data"), # Store selection
        Input("run-jobs-table", "selected_rows"),
        Input("run-jobs-table", "data"),
        prevent_initial_call=True
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
        Input("url", "pathname"), # Auto-refresh on page load
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
        Input("url", "pathname"),
        State("workflow-state", "data"),
        State("artifact-select", "options"),
    )(lambda path, state, opts: state.get("last_run_artifact") if (path == "/results" and state and "last_run_artifact" in state) else dash.no_update)

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
        prevent_initial_call=True
    )(_sync_path_preset)

    # 2. Run Artifact Path
    app.callback(
        Output("run-artifact-preset", "value"),
        Output("run-artifact-path", "value"),
        Input("run-artifact-preset", "value"),
        Input("run-artifact-path", "value"),
        prevent_initial_call=True
    )(_sync_path_preset)

    # 3. Results Artifact Root
    app.callback(
        Output("artifact-root-preset", "value"),
        Output("artifact-root-path", "value"),
        Input("artifact-root-preset", "value"),
        Input("artifact-root-path", "value"),
        prevent_initial_call=True
    )(_sync_path_preset)

    return app


# --- Module Scope Callbacks (Testing Logic) ---

def _cb_inspect_data(_n_clicks: int, upload_contents: str | None, upload_filename: str | None) -> tuple[Any, str, dict, str, str]:
    """Inspect data from uploaded file. Returns (result_div, error, state, file_label, file_path)."""
    final_path = ""
    display_name = ""
    
    if upload_contents:
        # Decode and save temp file
        import base64
        
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Save to a temporary location
        filename = upload_filename or "uploaded_data.csv"
        display_name = filename
        final_path = os.path.join("temp_data", filename)
        os.makedirs("temp_data", exist_ok=True)
        
        try:
            if "csv" in filename or "parquet" in filename:
                with open(final_path, 'wb') as f:
                    f.write(decoded)
            else:
                return None, "Unsupported file type (use .csv or .parquet)", {}, "No file selected", ""
        except Exception as e:
             return None, f"Upload failed: {e}", {}, "No file selected", ""
    else:
        # No upload — use bundled sample
        final_path = "examples/data/california_housing.csv"
        display_name = "california_housing.csv (sample)"
             
    if not final_path:
        return None, "", {}, "No file selected", ""
    
    result = inspect_data(final_path)
    if not result["success"]:
        return None, f"Error: {result.get('error')}", {}, "No file selected", ""
        
    stats_div = data_page.render_data_stats(result["stats"])
    preview_div = data_page.render_data_preview(result["preview"])
    
    label = f"✔ {display_name}  ({result['stats']['n_rows']} rows × {result['stats']['n_cols']} cols)"
    return html.Div([stats_div, preview_div]), "", {"data_path": final_path}, label, final_path


def _cb_save_target_col(target_col: str, state: dict) -> dict:
    if not state: 
        state = {}
    state["target_col"] = target_col
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
                if not isinstance(payload, dict): payload = {}
            except:
                payload = {}
            
            if "data" not in payload: payload["data"] = {}
            payload["data"]["path"] = d_path
            if t_col: 
                payload["data"]["target"] = t_col
            
            # Ensure minimal structure
            if "task" not in payload: payload["task"] = {"type": "regression"}
            if "config_version" not in payload: payload["config_version"] = 1
            
            new_yaml = yaml.safe_dump(payload, sort_keys=False)
            toast = make_toast(f"Imported settings from Data page.", icon="success")
            return new_yaml, "Imported data settings.", style_success, toast
            
        validate_config(yaml_text)
        toast = make_toast("Configuration is valid.", icon="success")
        return yaml_text, "Configuration is valid.", style_success, toast
    except Exception as exc:
        toast = make_toast(f"Error: {str(exc)}", icon="danger")
        return yaml_text, str(exc), style_error, toast


def _cb_handle_migration_preview(_n_clicks: int, input_path: str, target_ver: int) -> tuple[str, str, str | None]:
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


def _cb_update_tune_visibility(enabled: bool) -> dict:
    return {"display": "block"} if enabled else {"display": "none"}


def _cb_detect_run_action(yaml_text: str) -> tuple[str, str, str]:
    default_vals = ("fit", "Ready: TRAIN", "badge bg-primary fs-6 p-2 mb-3")
    if not yaml_text:
        return default_vals
        
    try:
        cfg = yaml.safe_load(yaml_text)
        if not isinstance(cfg, dict): return default_vals
        
        # Check Tuning
        if cfg.get("tuning", {}).get("enabled", False):
            return "tune", "Ready: TUNE", "badge bg-warning text-dark fs-6 p-2 mb-3"
            
        # Check Causal
        if "causal_method" in cfg.get("task", {}):
            return "fit", "Ready: CAUSAL ANALYSIS", "badge bg-info text-dark fs-6 p-2 mb-3"
            
        return default_vals
    except:
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


def _cb_populate_builder_options(pathname: str, state: dict | None) -> tuple[str, str, list, list, list, list, list, list]:
    if not state or "data_path" not in state:
        return "", "", [], [], [], [], [], []
        
    d_path = state.get("data_path", "")
    t_col = state.get("target_col", "")
    
    cols = []
    try:
        if d_path:
                res = inspect_data(d_path)
                if res["success"]:
                    cols = res["stats"]["columns"]
    except:
        pass
        
    # Exclude Columns Options: All columns except target
    non_target_opts = [{"label": c, "value": c} for c in cols if c != t_col]
    opts = [{"label": c, "value": c} for c in cols]
    target_opts = [{"label": c, "value": c} for c in cols]
    
    return d_path, t_col, target_opts, opts, opts, non_target_opts, opts, opts


def _cb_build_config_yaml(
    task_type, d_path, d_target, d_ids, d_cats, d_drops,
    s_type, s_n, s_seed, s_grp, s_time, s_ts_mode, s_test, s_gap, s_embargo,
    t_lr, t_leaves, t_est, t_depth, t_child, t_early, t_sub, t_col, t_l1, t_l2,
    tune_en, tune_pre, tune_tri, tune_obj,
    exp_dir,
    causal_en, causal_method,
    # Search Space Params
    tune_lr_min, tune_lr_max, tune_leaves_min, tune_leaves_max,
    tune_depth_min, tune_depth_max, tune_ff_min, tune_ff_max
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
            "lgb_params": {
                "learning_rate": t_lr,
                "num_leaves": t_leaves,
                "n_estimators": t_est,
                "max_depth": t_depth,
                "min_child_samples": t_child,
                "subsample": t_sub,
                "colsample_bytree": t_col,
                "reg_alpha": t_l1,
                "reg_lambda": t_l2,
            },
            "early_stopping_rounds": t_early,
            "seed": 42
        },
        "tuning": {"enabled": tune_en},
        "export": {"artifact_dir": exp_dir},
    }
    
    if causal_en and causal_method:
        cfg["task"]["causal_method"] = causal_method
        # Causal requires specific tasks usually, but we overlay it
        
    if d_ids: cfg["data"]["id_cols"] = d_ids
    if d_cats: cfg["data"]["categorical"] = d_cats
    if d_drops: cfg["data"]["drop_cols"] = d_drops
    
    if s_type == "group" and s_grp:
        cfg["split"]["group_col"] = s_grp
    if s_type == "timeseries":
        if s_time: cfg["split"]["time_col"] = s_time
        cfg["split"]["timeseries_mode"] = s_ts_mode
        if s_test: cfg["split"]["test_size"] = s_test
        cfg["split"]["gap"] = s_gap
        cfg["split"]["embargo"] = s_embargo
        
    if tune_en:
        cfg["tuning"]["preset"] = tune_pre
        cfg["tuning"]["n_trials"] = tune_tri
        if tune_obj: cfg["tuning"]["objective"] = tune_obj
        
        # Custom Search Space
        space = {}
        if tune_lr_min and tune_lr_max: space["learning_rate"] = {"low": float(tune_lr_min), "high": float(tune_lr_max), "log": True}
        if tune_leaves_min and tune_leaves_max: space["num_leaves"] = {"low": int(tune_leaves_min), "high": int(tune_leaves_max)}
        if tune_depth_min and tune_depth_max: space["max_depth"] = {"low": int(tune_depth_min), "high": int(tune_depth_max)}
        if tune_ff_min and tune_ff_max: space["feature_fraction"] = {"low": float(tune_ff_min), "high": float(tune_ff_max)}
        
        if space:
            cfg["tuning"]["search_space"] = space
        
    return yaml.dump(cfg, sort_keys=False)


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
        c_path = config_path_state or "configs/gui_run.yaml"
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


def _cb_refresh_run_jobs(_n_intervals: int, _refresh_clicks: int, last_status: dict | None, current_path: str, batch_mode: list) -> tuple[html.Div, Any, dict, str]:
    jobs = list_run_jobs(limit=100)
    
    new_status = {}
    toast = dash.no_update
    next_path = dash.no_update
    
    last_status = last_status or {}
    is_batch = "enabled" in (batch_mode or [])
    
    for job in jobs:
        new_status[job.job_id] = job.status
        
        old_s = last_status.get(job.job_id)
        if old_s in ["running", "queued"] and job.status in ["succeeded", "failed"]:
            icon = "success" if job.status == "succeeded" else "danger"
            toast = make_toast(f"Task {job.action} {job.status}!", icon=icon)
            
            if job.status == "succeeded" and not is_batch and current_path == "/run":
                    next_path = "/results"
                    # We can't update workflow-state here easily as it's not an output
                    # But the job result payload might have the artifact path.
                    # For now, we rely on the user finding it or future improvements to pass state.
                    pass
            
    data = [
        {
            "job_id": job.job_id,
            "action": job.action,
            "status": _status_badge(job.status),
            "created_at_utc": job.created_at_utc,
            "id": job.job_id
        }
        for job in jobs
    ]
    return task_table("run-jobs", data), toast, new_status, next_path


def _cb_show_selected_job_detail(selected_rows: list[int] | None, data: list[dict] | None) -> tuple[str, bool, str, str | None]:
    if not selected_rows or not data:
        return "Select a job to view details.", True, "", None
    
    row_idx = selected_rows[0]
    if row_idx >= len(data):
        return "Job not found.", True, "", None
        
    job_id = data[row_idx]["job_id"]
    job = get_run_job(job_id)
    
    if not job:
        return "Job details unavailable.", True, "", None
        
    can_cancel = job.status in ["queued", "running"]
    
    details_elems = [
        html.H5(f"Task: {job.action.upper()}", className="mb-3"),
        html.Div(
            [
                html.Span("Status: ", className="fw-bold"),
                html.Span(job.status.upper(), className=f"badge bg-{'success' if job.status=='succeeded' else 'danger' if job.status=='failed' else 'primary'}")
            ], 
            className="mb-2"
        ),
        html.Div([html.Span("Created: ", className="fw-bold"), html.Span(job.created_at_utc)], className="mb-2"),
        html.Div([html.Span("Started: ", className="fw-bold"), html.Span(job.started_at_utc or "-")], className="mb-2"),
        html.Div([html.Span("Finished: ", className="fw-bold"), html.Span(job.finished_at_utc or "-")], className="mb-2"),
    ]
    
    if job.error_message:
            details_elems.append(
                dbc.Alert(html.Pre(job.error_message, className="mb-0"), color="danger", className="mt-3")
            )
            
    if job.result and job.result.payload:
        payload_str = _json_dumps(job.result.payload)
        details_elems.append(html.Label("Result Payload", className="fw-bold mt-3"))
        details_elems.append(html.Pre(payload_str, className="bg-dark text-white p-2 rounded", style={"maxHeight": "300px", "overflow": "auto"}))
    
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


def _cb_list_artifacts(_n_clicks: int, _pathname: str, root_path: str) -> tuple[list[dict], list[dict]]:
    try:
        items = list_artifacts(root_path or "artifacts")
        options = [
            {
                "label": f"{item.created_at_utc} | {item.task_type} | {item.run_id}",
                "value": item.path,
            }
            for item in items
        ]
        return options, options
    except Exception:
        return [], []


def _cb_update_result_view(artifact_path: str | None, compare_path: str | None) -> tuple[Any, Any, Any, Any]:
    if not artifact_path:
        # Return empty placeholder charts with a message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#94a3b8", "family": "Inter"},
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "Select an artifact to view results",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5, "showarrow": False,
                "font": {"size": 16, "color": "#94a3b8"}
            }]
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

        kpi_elems = []
        metrics = art.metrics or {}
        for key in ["r2_score", "accuracy", "f1_score", "rmse", "mae"]:
            if key in metrics:
                val = metrics[key]
                kpi_elems.append(kpi_card(key, val))
        
        for k, v in metrics.items():
            if k not in ["r2_score", "accuracy", "f1_score", "rmse", "mae"] and isinstance(v, (int, float)):
                kpi_elems.append(kpi_card(k, v))
        
        kpi_container = html.Div(kpi_elems, className="d-flex flex-wrap gap-3")

        if comp_art:
            fig_main = plot_comparison_bar(
                art.metrics or {}, 
                comp_art.metrics or {}, 
                name1=f"Current ({art.run_id[:6]}...)", 
                name2=f"Baseline ({comp_art.run_id[:6]}...)"
            )
        else:
            fig_main = plot_metrics_bar(art.metrics or {}, title="Performance Metrics")

        fig_sec = {}
        if art.metadata and "feature_importance" in art.metadata:
            fig_sec = plot_feature_importance(art.metadata["feature_importance"])

        dt = str(art.created_at_utc) if art.created_at_utc else "n/a"
        details_str = (
            f"Run ID: {art.run_id}\n"
            f"Type: {art.task_type}\n"
            f"Created: {dt}\n\n"

            f"Run Config:\n{json.dumps(asdict(art.config) if is_dataclass(art.config) else art.config, indent=2)}"
        )
        details_elem = html.Pre(details_str, className="p-3 border rounded")
        
        # Safe default for figure
        if not fig_sec:
             fig_sec = go.Figure()
             fig_sec.update_layout(
                xaxis={"visible": False}, yaxis={"visible": False},
                annotations=[{"text": "No Feature Importance Available", "showarrow": False, "font": {"color": "white"}}]
             )

        return kpi_container, fig_main, fig_sec, details_elem

    except Exception as exc:
        return html.Div(f"Error loading artifact: {exc}"), {}, {}, ""


def _cb_evaluate_artifact_action(_n_clicks: int, artifact_path: str, data_path: str) -> str:
    try:
        if not artifact_path or not data_path:
            return "Artifact and Data path are required."
        
        artifact = Artifact.load(artifact_path)
        frame = load_tabular_data(data_path)
        result = evaluate(artifact, frame)
        
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
