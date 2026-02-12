"""Config editor page."""

from dash import dcc, html


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Config Editor"),
            html.Label("RunConfig YAML"),
            dcc.Textarea(
                id="config-yaml",
                style={"width": "100%", "height": "360px", "fontFamily": "monospace"},
                value=(
                    "config_version: 1\n"
                    "task:\n"
                    "  type: regression\n"
                    "data:\n"
                    "  path: examples/data/california_housing.csv\n"
                    "  target: MedHouseVal\n"
                    "split:\n"
                    "  type: kfold\n"
                    "  n_splits: 5\n"
                    "  seed: 42\n"
                    "export:\n"
                    "  artifact_dir: artifacts\n"
                ),
            ),
            html.Br(),
            html.Label("Config file path"),
            dcc.Input(
                id="config-file-path",
                type="text",
                style={"width": "100%"},
                value="configs/gui_run.yaml",
            ),
            html.Br(),
            html.Br(),
            html.Button("Validate", id="config-validate-btn", n_clicks=0),
            html.Button("Load", id="config-load-btn", n_clicks=0, style={"marginLeft": "8px"}),
            html.Button("Save", id="config-save-btn", n_clicks=0, style={"marginLeft": "8px"}),
            html.Pre(
                id="config-validation-result",
                style={"marginTop": "12px", "whiteSpace": "pre-wrap"},
            ),
        ]
    )
