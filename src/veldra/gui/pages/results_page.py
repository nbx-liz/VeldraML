"""Results page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Results & Artifacts", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Export Excel",
                            id="result-export-excel-btn",
                            color="secondary",
                            className="me-2",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Export HTML Report",
                            id="result-export-html-btn",
                            color="secondary",
                            className="me-2",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Download Config",
                            id="result-download-config-btn",
                            color="info",
                            outline=True,
                        ),
                        width="auto",
                    ),
                ],
                className="mb-3 g-2",
            ),
            html.Div(id="result-export-status", className="small text-info mb-3"),
            dcc.Download(id="result-config-download"),
            dcc.Download(id="result-report-download"),
            dcc.Store(id="result-export-job-store"),
            dcc.Interval(
                id="result-export-poll-interval",
                interval=1000,
                n_intervals=0,
                disabled=True,
            ),
            # Artifact Selection
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Artifact Root", className="fw-bold mb-2"),
                                    dbc.InputGroup(
                                        [
                                            dbc.Select(
                                                id="artifact-root-preset",
                                                options=[
                                                    {
                                                        "label": "Standard (artifacts/)",
                                                        "value": "artifacts",
                                                    },
                                                    {
                                                        "label": "Output (output/)",
                                                        "value": "output",
                                                    },
                                                    {"label": "Custom...", "value": "custom"},
                                                ],
                                                placeholder="Preset",
                                                style={"maxWidth": "100px"},
                                            ),
                                            dbc.Input(id="artifact-root-path", value="artifacts"),
                                            dbc.Button(
                                                "Refresh",
                                                id="artifact-refresh-btn",
                                                color="secondary",
                                            ),
                                        ]
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Select Artifact", className="fw-bold mb-2"
                                                ),
                                                dbc.Select(
                                                    id="artifact-select",
                                                    placeholder="Select a run artifact...",
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label(
                                                    "Compare With (Optional)",
                                                    className="fw-bold mb-2",
                                                ),
                                                dbc.Select(
                                                    id="artifact-select-compare",
                                                    placeholder="Select baseline...",
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ],
                                    className="g-2",  # Add a small gutter between dropdowns
                                ),
                                width=8,
                            ),
                        ]
                    )
                ],
                className="glass-card mb-4",
            ),
            dcc.Loading(
                id="results-loading",
                type="dot",
                children=html.Div(
                    [
                        # KPI Cards
                        html.Div(id="artifact-kpi-container", className="mb-4"),
                        # Charts & Details
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Tabs(
                                            [
                                                dbc.Tab(
                                                    dcc.Graph(
                                                        id="result-chart-main",
                                                        style={"height": "400px"},
                                                    ),
                                                    label="Overview",
                                                    tab_id="tab-chart-main",
                                                ),
                                                dbc.Tab(
                                                    dcc.Graph(
                                                        id="result-chart-secondary",
                                                        style={"height": "400px"},
                                                    ),
                                                    label="Feature Importance",
                                                    tab_id="tab-chart-secondary",
                                                ),
                                                dbc.Tab(
                                                    dcc.Graph(
                                                        id="result-learning-curve",
                                                        style={"height": "400px"},
                                                    ),
                                                    label="Learning Curves",
                                                    tab_id="tab-learning-curves",
                                                ),
                                                dbc.Tab(
                                                    html.Pre(
                                                        id="result-config-view",
                                                        style={
                                                            "height": "400px",
                                                            "overflowY": "auto",
                                                        },
                                                    ),
                                                    label="Config",
                                                    tab_id="tab-config",
                                                ),
                                            ],
                                            className="nav-fill mb-3",
                                        )
                                    ],
                                    width=8,
                                    className="glass-card p-3",
                                    style={"minHeight": "450px"},
                                ),
                                dbc.Col(
                                    [
                                        html.H5("Actions", className="mb-3"),
                                        dbc.Button(
                                            "Re-evaluate",
                                            id="artifact-evaluate-btn",
                                            color="primary",
                                            className="w-100 mb-3",
                                        ),
                                        html.Label(
                                            "Evaluation Data Path", className="small text-muted"
                                        ),
                                        dbc.Input(
                                            id="artifact-eval-data-path",
                                            placeholder="path/to/data.csv",
                                            className="mb-3",
                                        ),
                                        html.Hr(),
                                        html.H5("Overview", className="mb-2"),
                                        html.Div(id="result-overview-summary", className="mb-3"),
                                        html.H5("Details", className="mb-2"),
                                        html.Div(id="result-details"),
                                        html.Pre(
                                            id="artifact-eval-result",
                                            style={
                                                "height": "250px",
                                                "overflowY": "auto",
                                                "backgroundColor": "#0d0d0d",
                                                "fontSize": "12px",
                                                "border": "1px solid rgba(148, 163, 184, 0.1)",
                                                "borderRadius": "4px",
                                                "padding": "8px",
                                            },
                                        ),
                                    ],
                                    width=4,
                                    className="glass-card p-4 ms-3",
                                ),
                            ],
                            className="g-0",  # gutter 0 for tighter layout
                        ),
                    ]
                ),
            ),
        ]
    )
