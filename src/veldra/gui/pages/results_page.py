"""Results page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from veldra.gui.components.guided_mode_banner import guided_mode_banner


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Results & Artifacts", className="mb-4"),
            guided_mode_banner("results"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Export Excel",
                            id="result-export-excel-btn",
                            color="secondary",
                            className="me-2 result-export-btn",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Export HTML Report",
                            id="result-export-html-btn",
                            color="secondary",
                            className="me-2 result-export-btn",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Export PDF Report",
                            id="result-export-pdf-btn",
                            color="secondary",
                            className="me-2 result-export-btn",
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
            html.Div(id="result-export-help", className="small text-muted mb-3"),
            dcc.Download(id="result-config-download"),
            dcc.Download(id="result-report-download"),
            dcc.Store(id="result-export-job-store"),
            dcc.Interval(
                id="result-export-poll-interval",
                interval=1000,
                n_intervals=0,
                disabled=True,
            ),
            dcc.Store(id="artifact-page", data=0),
            dcc.Store(id="artifact-page-size", data=50),
            dcc.Store(id="artifact-total", data=0),
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
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Input(
                                                    id="artifact-search",
                                                    placeholder="Search run_id / task_type",
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dbc.Select(
                                                    id="artifact-page-size-select",
                                                    options=[
                                                        {"label": "25", "value": 25},
                                                        {"label": "50", "value": 50},
                                                        {"label": "100", "value": 100},
                                                    ],
                                                    value=50,
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    id="artifact-page-info",
                                                    className="small text-muted mt-2",
                                                ),
                                                width=2,
                                            ),
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        dbc.Button(
                                                            "Prev",
                                                            id="artifact-page-prev-btn",
                                                            size="sm",
                                                            color="secondary",
                                                            outline=True,
                                                            className="me-1",
                                                        ),
                                                        dbc.Button(
                                                            "Next",
                                                            id="artifact-page-next-btn",
                                                            size="sm",
                                                            color="secondary",
                                                            outline=True,
                                                        ),
                                                    ]
                                                ),
                                                width=2,
                                            ),
                                        ],
                                        className="g-2 mt-2",
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
                                                    dcc.Graph(
                                                        id="result-fold-metrics",
                                                        style={"height": "400px"},
                                                    ),
                                                    label="Fold Metrics",
                                                    tab_id="tab-fold-metrics",
                                                ),
                                                dbc.Tab(
                                                    dcc.Graph(
                                                        id="result-causal-diagnostics",
                                                        style={"height": "400px"},
                                                    ),
                                                    label="Causal Diagnostics",
                                                    tab_id="tab-causal-diagnostics",
                                                ),
                                                dbc.Tab(
                                                    html.Div(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dcc.Dropdown(
                                                                            id="result-drilldown-feature",
                                                                            options=[],
                                                                            placeholder=(
                                                                                "Select feature"
                                                                            ),
                                                                        ),
                                                                        width=8,
                                                                    ),
                                                                    dbc.Col(
                                                                        dbc.Input(
                                                                            id="result-drilldown-topn",
                                                                            type="number",
                                                                            min=1,
                                                                            max=100,
                                                                            step=1,
                                                                            value=20,
                                                                        ),
                                                                        width=4,
                                                                    ),
                                                                ],
                                                                className="mb-2",
                                                            ),
                                                            dcc.Graph(
                                                                id="result-feature-drilldown",
                                                                style={"height": "330px"},
                                                            ),
                                                        ]
                                                    ),
                                                    label="Feature Drilldown",
                                                    tab_id="tab-feature-drilldown",
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
                                        html.Div(
                                            (
                                                "Re-evaluate checks model performance "
                                                "on another dataset."
                                            ),
                                            className="small text-muted mb-2",
                                        ),
                                        html.Label(
                                            "Evaluation Data Path", className="small text-muted"
                                        ),
                                        dbc.Input(
                                            id="artifact-eval-data-path",
                                            placeholder="path/to/data.csv",
                                            className="mb-3",
                                        ),
                                        html.Div(id="artifact-eval-precheck", className="mb-3"),
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
