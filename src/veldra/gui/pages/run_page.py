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
            html.Button("Run", id="run-execute-btn", n_clicks=0),
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
        ]
    )
