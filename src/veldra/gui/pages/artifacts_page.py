"""Artifact explorer page."""

from dash import dcc, html


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Artifact Explorer"),
            html.Label("Artifact root directory"),
            dcc.Input(
                id="artifact-root-path",
                type="text",
                style={"width": "100%"},
                value="artifacts",
            ),
            html.Br(),
            html.Br(),
            html.Button("Refresh", id="artifact-refresh-btn", n_clicks=0),
            html.Br(),
            html.Br(),
            html.Label("Artifacts"),
            dcc.Dropdown(id="artifact-select", options=[], value=None),
            html.Br(),
            html.Div(id="artifact-metrics"),
            html.Hr(),
            html.H4("Evaluate selected artifact"),
            html.Label("Evaluation data path"),
            dcc.Input(
                id="artifact-eval-data-path",
                type="text",
                style={"width": "100%"},
                value="",
            ),
            html.Br(),
            html.Br(),
            html.Button("Evaluate", id="artifact-evaluate-btn", n_clicks=0),
            html.Pre(
                id="artifact-eval-result",
                style={"marginTop": "12px", "whiteSpace": "pre-wrap"},
            ),
        ]
    )
