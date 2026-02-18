"""Run console page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from veldra.gui.components.task_table import task_table


def layout(state: dict | None = None) -> html.Div:
    data_path_val = ""
    config_path_val = "configs/gui_run.yaml"  # Default
    config_yaml_val = ""

    if state:
        if "data_path" in state:
            data_path_val = state["data_path"]
        if "config_yaml" in state:
            config_yaml_val = state["config_yaml"] or ""
        # If we had a mechanism to store last saved config path, we'd load it here.
        # identifying readiness

    # Readiness Check (Simple)
    is_ready = bool(data_path_val)
    # Config is assumed to be handled by builder which saves to default path.

    actions = ["fit", "evaluate", "tune", "simulate", "export", "estimate_dr"]

    return html.Div(
        [
            html.H2("Run Tasks", className="mb-4"),
            # Action Selection & Summary
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5("1. Auto-Action Mode", className="text-info mb-3"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.RadioItems(
                                                    id="run-action-override-mode",
                                                    options=[
                                                        {"label": "Auto", "value": "auto"},
                                                        {"label": "Manual", "value": "manual"},
                                                    ],
                                                    value="auto",
                                                    inline=True,
                                                ),
                                                width=12,
                                            ),
                                            dbc.Col(
                                                dbc.Select(
                                                    id="run-action-manual",
                                                    options=[
                                                        {"label": a.upper(), "value": a}
                                                        for a in actions
                                                    ],
                                                    value="fit",
                                                ),
                                                width=12,
                                                className="mt-2",
                                                id="run-action-manual-container",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        [
                                            dbc.RadioItems(
                                                id="run-action",
                                                options=[
                                                    {"label": a.upper(), "value": a}
                                                    for a in actions
                                                ],
                                                value="fit",
                                                inline=True,
                                                style={"display": "none"},
                                            ),
                                            html.Div(
                                                id="run-action-display",
                                                className="badge bg-primary fs-6 p-2 mb-3",
                                                children="Ready: TRAIN",
                                            ),
                                            html.Div(
                                                id="run-action-description",
                                                className="small text-muted mb-2",
                                            ),
                                        ]
                                    ),
                                    html.Hr(className="border-secondary"),
                                    html.H5("2. Task Context", className="text-info mb-3"),
                                    # Hidden Inputs for Auto-Filling
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Config Source"),
                                                    dbc.Input(
                                                        id="run-config-path",
                                                        value=config_path_val,
                                                        readonly=True,
                                                        className="mb-2 bg-dark text-muted",
                                                    ),
                                                ],
                                                width=12,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Data Source"),
                                                    dbc.Input(
                                                        id="run-data-path",
                                                        value=data_path_val,
                                                        readonly=True,
                                                        className="mb-2 bg-dark text-muted",
                                                    ),
                                                ],
                                                width=12,
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Accordion(
                                        [
                                            dbc.AccordionItem(
                                                [
                                                    html.Label("Artifact Path"),
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.Select(
                                                                id="run-artifact-preset",
                                                                options=[
                                                                    {
                                                                        "label": "Standard "
                                                                        "(artifacts/)",
                                                                        "value": "artifacts",
                                                                    },
                                                                    {
                                                                        "label": "Output (output/)",
                                                                        "value": "output",
                                                                    },
                                                                    {
                                                                        "label": "Custom...",
                                                                        "value": "custom",
                                                                    },
                                                                ],
                                                                placeholder="Preset",
                                                                style={"maxWidth": "120px"},
                                                            ),
                                                            dbc.Input(
                                                                id="run-artifact-path",
                                                                value="artifacts",
                                                                placeholder="e.g. artifacts",
                                                            ),
                                                        ],
                                                        className="mb-2",
                                                    ),
                                                    html.Label("Scenarios Path"),
                                                    dbc.Input(
                                                        id="run-scenarios-path", className="mb-2"
                                                    ),
                                                    html.Label("Export Format"),
                                                    dbc.Select(
                                                        id="run-export-format",
                                                        options=[
                                                            {"label": "Python", "value": "python"},
                                                            {"label": "ONNX", "value": "onnx"},
                                                        ],
                                                        value="python",
                                                    ),
                                                    html.Label("Queue Priority", className="mt-2"),
                                                    dbc.Select(
                                                        id="run-priority",
                                                        options=[
                                                            {"label": "High", "value": "high"},
                                                            {"label": "Normal", "value": "normal"},
                                                            {"label": "Low", "value": "low"},
                                                        ],
                                                        value="normal",
                                                    ),
                                                    html.Label(
                                                        "Config YAML Override", className="mt-2"
                                                    ),
                                                    dcc.Textarea(
                                                        id="run-config-yaml",
                                                        value=config_yaml_val,
                                                        style={
                                                            "height": "100px",
                                                            "width": "100%",
                                                            "fontFamily": "monospace",
                                                        },
                                                        className="form-control",
                                                    ),
                                                ],
                                                title="Advanced Options (Override)",
                                            )
                                        ],
                                        start_collapsed=True,
                                        className="mb-4",
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                [
                                                    html.I(className="bi bi-play-fill me-2"),
                                                    "Launch Task",
                                                ],
                                                id="run-execute-btn",
                                                color="primary",
                                                size="lg",
                                                className="w-100 shadow-lg",
                                                disabled=not is_ready,
                                                title="Data must be selected in Data Page"
                                                if not is_ready
                                                else "Include Config from Builder",
                                            ),
                                            dbc.Alert(
                                                "Ready: Data source is set."
                                                if is_ready
                                                else (
                                                    "Not ready: Select and inspect "
                                                    "a data file in Data page."
                                                ),
                                                id="run-launch-status",
                                                color="success" if is_ready else "warning",
                                                className="mt-2 mb-0 small",
                                            ),
                                        ],
                                        id="run-launch-container",
                                    ),
                                    html.Div(id="run-guardrail-container", className="mt-3"),
                                    dcc.Store(id="run-guardrail-has-error", data=False),
                                    dcc.Loading(
                                        id="run-loading",
                                        type="dot",
                                        children=html.Div(
                                            id="run-result-log", className="mt-3 text-info small"
                                        ),
                                    ),
                                ],
                                className="glass-card h-100",
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            # Live Task Feed
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H5(
                                                "3. Execution Queue", className="text-info mb-0"
                                            ),
                                            html.Div(
                                                [
                                                    # Batch Mode Toggle (Placeholder for future)
                                                    dbc.Checklist(
                                                        options=[
                                                            {
                                                                "label": "Batch Mode",
                                                                "value": "enabled",
                                                            }
                                                        ],
                                                        value=[],
                                                        id="run-batch-mode-toggle",
                                                        switch=True,
                                                        inline=True,
                                                        className="me-3 d-inline-block small",
                                                    ),
                                                    dbc.Button(
                                                        "Refresh",
                                                        id="run-refresh-jobs-btn",
                                                        size="sm",
                                                        color="secondary",
                                                        outline=True,
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className=(
                                            "d-flex justify-content-between align-items-center mb-3"
                                        ),
                                    ),
                                    dcc.Interval(
                                        id="run-jobs-interval", interval=2000, n_intervals=0
                                    ),
                                    html.Div(
                                        id="run-jobs-table-container",
                                        children=task_table("run-jobs", []),  # Initial empty table
                                        style={"minHeight": "200px"},
                                    ),
                                    html.Hr(className="border-secondary my-4"),
                                    html.H5("Task Details", className="mb-3"),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Cancel Task",
                                                id="run-cancel-job-btn",
                                                color="danger",
                                                size="sm",
                                                className="me-2",
                                                disabled=True,
                                            ),
                                            html.Span(
                                                id="selected-job-id-display",
                                                className="text-muted small",
                                            ),
                                            # Link to Results
                                            dbc.Button(
                                                "View Results →",
                                                id="run-view-results-btn",
                                                color="success",
                                                size="sm",
                                                outline=True,
                                                href="/results",
                                                className="float-end",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="run-queue-priority",
                                                options=[
                                                    {"label": "High", "value": "high"},
                                                    {"label": "Normal", "value": "normal"},
                                                    {"label": "Low", "value": "low"},
                                                ],
                                                value="normal",
                                            ),
                                            dbc.Button(
                                                "Set Priority",
                                                id="run-set-priority-btn",
                                                color="secondary",
                                                size="sm",
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    dbc.Button(
                                        "Load More Logs",
                                        id="run-log-load-more-btn",
                                        color="secondary",
                                        size="sm",
                                        className="mb-2",
                                    ),
                                    html.Div(id="run-job-detail", className="small"),
                                    # Hidden store for selection state
                                    dcc.Store(id="run-job-select"),
                                    dcc.Store(id="run-log-limit", data=200),
                                ],
                                className="glass-card h-100",
                            )
                        ],
                        width=8,
                    ),
                ]
            ),
        ]
    )
