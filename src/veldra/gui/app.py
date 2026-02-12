"""Dash app factory for Veldra GUI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html

from veldra.api.artifact import Artifact
from veldra.api.runner import evaluate
from veldra.data import load_tabular_data
from veldra.gui.pages import artifacts_page, config_page, run_page
from veldra.gui.services import (
    cancel_run_job,
    get_run_job,
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


def _sidebar() -> html.Div:
    return html.Div(
        [
            html.H3("Veldra GUI"),
            dbc.Nav(
                [
                    dbc.NavLink("Config", href="/config", active="exact"),
                    dbc.NavLink("Run", href="/run", active="exact"),
                    dbc.NavLink("Artifacts", href="/artifacts", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style={
            "padding": "16px",
            "borderRight": "1px solid #d9d9d9",
            "minHeight": "100vh",
        },
    )


def _main_layout() -> html.Div:
    return html.Div(
        [
            dcc.Location(id="url"),
            dbc.Row(
                [
                    dbc.Col(_sidebar(), width=3),
                    dbc.Col(
                        [
                            html.Div(
                                "RunConfig-driven local operations GUI (MVP).",
                                id="gui-global-message",
                                style={"marginTop": "12px"},
                            ),
                            html.Hr(),
                            html.Div(id="page-content"),
                        ],
                        width=9,
                    ),
                ],
                style={"margin": "0"},
            ),
        ]
    )


def render_page(pathname: str | None) -> Any:
    if pathname == "/config":
        return config_page.layout()
    if pathname == "/run":
        return run_page.layout()
    if pathname == "/artifacts":
        return artifacts_page.layout()
    return config_page.layout()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def handle_config_action(
    triggered: str,
    yaml_text: str | None,
    config_path: str | None,
) -> tuple[str, str]:
    current_yaml = yaml_text or ""
    try:
        if triggered == "config-load-btn":
            loaded = load_config_yaml(config_path or "")
            return loaded, f"Loaded: {config_path}"
        if triggered == "config-save-btn":
            saved_path = save_config_yaml(config_path or "", current_yaml)
            return current_yaml, f"Saved: {saved_path}"
        parsed = validate_config(current_yaml)
        return (
            current_yaml,
            f"Valid RunConfig: task={parsed.task.type}, target={parsed.data.target}",
        )
    except Exception as exc:
        return current_yaml, str(exc)


def handle_config_migrate_preview(yaml_text: str, target_version: int) -> tuple[str, str, str]:
    normalized_yaml, diff, result = migrate_config_from_yaml(
        yaml_text or "",
        target_version=int(target_version),
    )
    return normalized_yaml, diff or "(no diff)", _json_dumps(result)


def handle_config_migrate_apply(
    input_path: str,
    output_path: str,
    target_version: int,
) -> str:
    result = migrate_config_file_via_gui(
        input_path=input_path,
        output_path=(output_path.strip() if output_path and output_path.strip() else None),
        target_version=int(target_version),
    )
    return _json_dumps(result)


def format_run_action_result(
    action: str,
    config_yaml: str,
    config_path: str,
    data_path: str,
    artifact_path: str,
    scenarios_path: str,
    export_format: str,
) -> tuple[str, str]:
    result = run_action(
        RunInvocation(
            action=action or "",
            config_yaml=config_yaml,
            config_path=config_path,
            data_path=data_path,
            artifact_path=artifact_path,
            scenarios_path=scenarios_path,
            export_format=export_format,
        )
    )
    payload = _json_dumps(result.payload) if result.payload else "{}"
    status = "SUCCESS" if result.success else "ERROR"
    return payload, f"[{status}] {result.message}"


def enqueue_run_job_result(
    action: str,
    config_yaml: str,
    config_path: str,
    data_path: str,
    artifact_path: str,
    scenarios_path: str,
    export_format: str,
) -> tuple[str, str]:
    result = submit_run_job(
        RunInvocation(
            action=action or "",
            config_yaml=config_yaml,
            config_path=config_path,
            data_path=data_path,
            artifact_path=artifact_path,
            scenarios_path=scenarios_path,
            export_format=export_format,
        )
    )
    payload = _json_dumps(asdict(result))
    return payload, f"[QUEUED] {result.message}"


def build_artifact_options(root_dir: str | None) -> tuple[list[dict[str, str]], str | None]:
    items = list_artifacts(root_dir or "artifacts")
    options = [
        {
            "label": f"{item.run_id} | {item.task_type} | {item.created_at_utc or 'n/a'}",
            "value": item.path,
        }
        for item in items
    ]
    value = options[0]["value"] if options else None
    return options, value


def build_job_options(limit: int = 50) -> tuple[list[dict[str, str]], str | None, str]:
    jobs = list_run_jobs(limit=limit)
    options = [
        {
            "label": f"{job.job_id[:8]} | {job.action} | {job.status} | {job.created_at_utc}",
            "value": job.job_id,
        }
        for job in jobs
    ]
    value = options[0]["value"] if options else None
    table_payload = [
        {
            "job_id": job.job_id,
            "action": job.action,
            "status": job.status,
            "cancel_requested": job.cancel_requested,
            "created_at_utc": job.created_at_utc,
            "updated_at_utc": job.updated_at_utc,
        }
        for job in jobs
    ]
    return options, value, _json_dumps(table_payload)


def format_job_detail(job_id: str | None) -> str:
    if job_id is None or not job_id.strip():
        return "No job selected."
    job = get_run_job(job_id.strip())
    if job is None:
        return f"Job not found: {job_id}"
    payload = asdict(job)
    return _json_dumps(payload)


def format_artifact_metrics(artifact_path: str | None) -> Any:
    if artifact_path is None or not artifact_path.strip():
        return html.Pre("No artifact selected.")
    try:
        artifact = Artifact.load(artifact_path)
        payload = {
            "run_id": artifact.manifest.run_id,
            "task_type": artifact.run_config.task.type,
            "metrics": artifact.metrics.get("mean", artifact.metrics),
        }
        return html.Pre(_json_dumps(payload))
    except Exception as exc:
        return html.Pre(str(exc))


def evaluate_selected_artifact(artifact_path: str, data_path: str) -> str:
    try:
        artifact = Artifact.load(artifact_path)
        frame = load_tabular_data(data_path)
        result = evaluate(artifact, frame)
        payload = asdict(result)
        if "data" in payload and hasattr(payload["data"], "to_dict"):
            payload["data"] = payload["data"].head(20).to_dict(orient="records")
        return _json_dumps(payload)
    except Exception as exc:
        return str(exc)


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="Veldra GUI",
    )
    app.layout = _main_layout()
    poll_ms = max(200, int(os.getenv("VELDRA_GUI_POLL_MS", "2000")))

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def _render_page(pathname: str | None) -> Any:
        return render_page(pathname)

    @app.callback(
        Output("config-yaml", "value"),
        Output("config-validation-result", "children"),
        Input("config-validate-btn", "n_clicks"),
        Input("config-load-btn", "n_clicks"),
        Input("config-save-btn", "n_clicks"),
        State("config-yaml", "value"),
        State("config-file-path", "value"),
        prevent_initial_call=True,
    )
    def _handle_config_actions(
        _validate_clicks: int,
        _load_clicks: int,
        _save_clicks: int,
        yaml_text: str,
        config_path: str,
    ) -> tuple[str, str]:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        return handle_config_action(triggered, yaml_text, config_path)

    @app.callback(
        Output("config-migrate-normalized-yaml", "value"),
        Output("config-migrate-diff", "children"),
        Output("config-migrate-result", "children"),
        Input("config-migrate-preview-btn", "n_clicks"),
        Input("config-migrate-apply-btn", "n_clicks"),
        State("config-yaml", "value"),
        State("config-migrate-input-path", "value"),
        State("config-migrate-output-path", "value"),
        State("config-migrate-target-version", "value"),
        State("config-migrate-normalized-yaml", "value"),
        State("config-migrate-diff", "children"),
        prevent_initial_call=True,
    )
    def _handle_migrate_actions(
        _preview_clicks: int,
        _apply_clicks: int,
        yaml_text: str,
        input_path: str,
        output_path: str,
        target_version: int,
        normalized_existing: str,
        diff_existing: str,
    ) -> tuple[str, str, str]:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        try:
            if triggered == "config-migrate-preview-btn":
                return handle_config_migrate_preview(yaml_text, int(target_version or 1))
            message = handle_config_migrate_apply(
                input_path=input_path or "",
                output_path=output_path or "",
                target_version=int(target_version or 1),
            )
            return normalized_existing or "", diff_existing or "", message
        except Exception as exc:
            return (
                normalized_existing or "",
                diff_existing or "",
                normalize_gui_error(exc),
            )

    @app.callback(
        Output("run-jobs-interval", "interval"),
        Input("url", "pathname"),
    )
    def _set_run_polling(_pathname: str | None) -> int:
        return poll_ms

    @app.callback(
        Output("run-result-json", "children"),
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
    )
    def _enqueue_run_action(
        _n_clicks: int,
        action: str,
        config_yaml: str,
        config_path: str,
        data_path: str,
        artifact_path: str,
        scenarios_path: str,
        export_format: str,
    ) -> tuple[str, str]:
        try:
            return enqueue_run_job_result(
                action=action,
                config_yaml=config_yaml,
                config_path=config_path,
                data_path=data_path,
                artifact_path=artifact_path,
                scenarios_path=scenarios_path,
                export_format=export_format,
            )
        except Exception as exc:
            return "{}", f"[ERROR] {normalize_gui_error(exc)}"

    @app.callback(
        Output("run-job-select", "options"),
        Output("run-job-select", "value"),
        Output("run-jobs-table", "children"),
        Input("run-jobs-interval", "n_intervals"),
        Input("run-refresh-jobs-btn", "n_clicks"),
        State("run-job-select", "value"),
    )
    def _refresh_run_jobs(
        _n_intervals: int,
        _refresh_clicks: int,
        selected_job_id: str | None,
    ) -> tuple[list[dict[str, str]], str | None, str]:
        options, default_value, table_text = build_job_options(limit=100)
        if selected_job_id is not None and any(opt["value"] == selected_job_id for opt in options):
            return options, selected_job_id, table_text
        return options, default_value, table_text

    @app.callback(
        Output("run-job-detail", "children"),
        Input("run-job-select", "value"),
        Input("run-jobs-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def _show_job_detail(job_id: str | None, _n_intervals: int) -> str:
        return format_job_detail(job_id)

    @app.callback(
        Output("run-result-log", "children", allow_duplicate=True),
        Input("run-cancel-job-btn", "n_clicks"),
        State("run-job-select", "value"),
        prevent_initial_call=True,
    )
    def _cancel_job(_n_clicks: int, job_id: str | None) -> str:
        try:
            if job_id is None or not job_id.strip():
                return "[ERROR] No job selected."
            canceled = cancel_run_job(job_id.strip())
            return f"[INFO] {canceled.message}"
        except Exception as exc:
            return f"[ERROR] {normalize_gui_error(exc)}"

    @app.callback(
        Output("artifact-select", "options"),
        Output("artifact-select", "value"),
        Input("artifact-refresh-btn", "n_clicks"),
        State("artifact-root-path", "value"),
        prevent_initial_call=True,
    )
    def _refresh_artifacts(
        _n_clicks: int,
        root_dir: str,
    ) -> tuple[list[dict[str, str]], str | None]:
        return build_artifact_options(root_dir)

    @app.callback(
        Output("artifact-metrics", "children"),
        Input("artifact-select", "value"),
        prevent_initial_call=True,
    )
    def _show_artifact_metrics(artifact_path: str | None) -> Any:
        return format_artifact_metrics(artifact_path)

    @app.callback(
        Output("artifact-eval-result", "children"),
        Input("artifact-evaluate-btn", "n_clicks"),
        State("artifact-select", "value"),
        State("artifact-eval-data-path", "value"),
        prevent_initial_call=True,
    )
    def _evaluate_artifact(_n_clicks: int, artifact_path: str, data_path: str) -> str:
        return evaluate_selected_artifact(artifact_path, data_path)

    return app
