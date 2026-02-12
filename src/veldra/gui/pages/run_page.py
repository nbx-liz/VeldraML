"""Run console page."""

from dash import dcc, html


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Run Console"),
            html.Label("Action"),
            dcc.Dropdown(
                id="run-action",
                options=[
                    {"label": "fit", "value": "fit"},
                    {"label": "evaluate", "value": "evaluate"},
                    {"label": "tune", "value": "tune"},
                    {"label": "simulate", "value": "simulate"},
                    {"label": "export", "value": "export"},
                    {"label": "estimate_dr", "value": "estimate_dr"},
                ],
                value="fit",
                clearable=False,
            ),
            html.Br(),
            html.Label("Config YAML (optional if config path is provided)"),
            dcc.Textarea(
                id="run-config-yaml",
                style={"width": "100%", "height": "220px", "fontFamily": "monospace"},
                value="",
            ),
            html.Br(),
            html.Label("Config path"),
            dcc.Input(id="run-config-path", type="text", style={"width": "100%"}, value=""),
            html.Br(),
            html.Br(),
            html.Label("Data path"),
            dcc.Input(id="run-data-path", type="text", style={"width": "100%"}, value=""),
            html.Br(),
            html.Br(),
            html.Label("Artifact path"),
            dcc.Input(id="run-artifact-path", type="text", style={"width": "100%"}, value=""),
            html.Br(),
            html.Br(),
            html.Label("Scenarios path"),
            dcc.Input(id="run-scenarios-path", type="text", style={"width": "100%"}, value=""),
            html.Br(),
            html.Br(),
            html.Label("Export format"),
            dcc.Dropdown(
                id="run-export-format",
                options=[
                    {"label": "python", "value": "python"},
                    {"label": "onnx", "value": "onnx"},
                ],
                value="python",
                clearable=False,
            ),
            html.Br(),
            html.Button("Enqueue Job", id="run-execute-btn", n_clicks=0),
            html.Button(
                "Refresh Jobs",
                id="run-refresh-jobs-btn",
                n_clicks=0,
                style={"marginLeft": "8px"},
            ),
            html.Button(
                "Cancel Selected Job",
                id="run-cancel-job-btn",
                n_clicks=0,
                style={"marginLeft": "8px"},
            ),
            dcc.Interval(id="run-jobs-interval", interval=2000, n_intervals=0),
            html.Pre(
                id="run-result-log",
                style={"marginTop": "12px", "whiteSpace": "pre-wrap"},
            ),
            html.Pre(
                id="run-result-json",
                style={
                    "marginTop": "12px",
                    "whiteSpace": "pre-wrap",
                    "maxHeight": "360px",
                    "overflowY": "auto",
                },
            ),
            html.Hr(),
            html.H4("Async Jobs"),
            dcc.Dropdown(id="run-job-select", options=[], value=None),
            html.Pre(
                id="run-jobs-table",
                style={
                    "marginTop": "12px",
                    "whiteSpace": "pre-wrap",
                    "maxHeight": "220px",
                    "overflowY": "auto",
                },
            ),
            html.Pre(
                id="run-job-detail",
                style={
                    "marginTop": "12px",
                    "whiteSpace": "pre-wrap",
                    "maxHeight": "220px",
                    "overflowY": "auto",
                },
            ),
        ]
    )
