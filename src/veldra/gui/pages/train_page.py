"""Train settings page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from veldra.gui.components.config_library import render_config_library
from veldra.gui.components.config_summary import render_config_summary
from veldra.gui.components.config_wizard import render_config_wizard
from veldra.gui.components.guided_mode_banner import guided_mode_banner
from veldra.gui.components.help_ui import help_icon
from veldra.gui.template_service import custom_slot_options, template_options


def layout(state: dict | None = None) -> html.Div:
    state = state or {}
    train = state.get("train_config") or {}
    tune = state.get("tuning_config") or {}
    yaml_text = state.get("config_yaml") or ""
    tpl_options = template_options()
    slot_options = custom_slot_options(state.get("custom_config_slots"))

    builder = html.Div(
        [
            render_config_library("train", template_options=tpl_options, slot_options=slot_options),
            render_config_wizard("train"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label("Learning Rate", className="fw-bold mb-0"),
                                    help_icon("train_learning_rate"),
                                ],
                                className="d-flex align-items-center gap-1 mb-1",
                            ),
                            dcc.Slider(
                                id="train-learning-rate",
                                min=0.001,
                                max=0.3,
                                step=0.001,
                                value=float(train.get("learning_rate", 0.05)),
                                marks={0.01: "0.01", 0.05: "0.05", 0.1: "0.1", 0.2: "0.2"},
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Num Boost Round", className="fw-bold"),
                            dbc.Input(
                                id="train-num-boost-round",
                                type="number",
                                value=int(train.get("num_boost_round", 300)),
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label("Num Leaves", className="fw-bold mb-0"),
                                    help_icon("train_num_leaves"),
                                ],
                                className="d-flex align-items-center gap-1 mb-1",
                            ),
                            dbc.Input(
                                id="train-num-leaves",
                                type="number",
                                value=int(train.get("num_leaves", 31)),
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Conservative",
                                    id="train-preset-conservative-btn",
                                    color="secondary",
                                    outline=True,
                                ),
                                dbc.Button(
                                    "Balanced",
                                    id="train-preset-balanced-btn",
                                    color="secondary",
                                    outline=True,
                                ),
                            ],
                            size="sm",
                        ),
                        width="auto",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Max Depth", className="mb-0"),
                                                    help_icon("train_max_depth"),
                                                ],
                                                className="d-flex align-items-center gap-1 mb-1",
                                            ),
                                            dbc.Input(
                                                id="train-max-depth",
                                                type="number",
                                                value=int(train.get("max_depth", -1)),
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Min Child Samples", className="mb-0"
                                                    ),
                                                    help_icon("train_min_child_samples"),
                                                ],
                                                className="d-flex align-items-center gap-1 mb-1",
                                            ),
                                            dbc.Input(
                                                id="train-min-child",
                                                type="number",
                                                value=int(train.get("min_child_samples", 20)),
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Early Stopping Rounds",
                                                        className="mb-0",
                                                    ),
                                                    help_icon("train_early_stopping"),
                                                ],
                                                className="d-flex align-items-center gap-1 mb-1",
                                            ),
                                            dbc.Input(
                                                id="train-early-stopping",
                                                type="number",
                                                value=int(train.get("early_stopping_rounds", 100)),
                                            ),
                                        ],
                                        width=4,
                                    ),
                                ]
                            ),
                            html.Hr(),
                            dbc.Switch(
                                id="train-auto-class-weight",
                                label="Auto Class Weight",
                                value=bool(train.get("auto_class_weight", True)),
                            ),
                            dbc.Input(
                                id="train-class-weight",
                                placeholder='{"0": 1.0, "1": 2.0}',
                                value=train.get("class_weight_text", ""),
                            ),
                        ],
                        title="Advanced Training",
                    ),
                    dbc.AccordionItem(
                        [
                            dbc.Switch(
                                id="train-tune-enabled",
                                label="Enable tuning",
                                value=bool(tune.get("enabled", False)),
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Select(
                                            id="train-tune-preset",
                                            options=[
                                                {"label": "Fast", "value": "fast"},
                                                {"label": "Standard", "value": "standard"},
                                            ],
                                            value=tune.get("preset", "standard"),
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="train-tune-trials",
                                            type="number",
                                            value=int(tune.get("n_trials", 30)),
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Select(
                                            id="train-tune-objective",
                                            options=[],
                                            value=tune.get("objective"),
                                        ),
                                        width=4,
                                    ),
                                ],
                                className="mt-2",
                            ),
                            html.Div(id="train-objective-help", className="mt-2"),
                        ],
                        title="Tuning",
                    ),
                ],
                start_collapsed=True,
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Artifact Directory", className="fw-bold"),
                            dbc.Input(
                                id="train-artifact-dir",
                                value=state.get("artifact_dir", "artifacts"),
                            ),
                        ],
                        width=6,
                    )
                ],
                className="mb-3",
            ),
            html.Div(id="train-guardrail-container", className="mb-3"),
            html.Div(id="train-summary-container", children=render_config_summary(state)),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Back: Validation", href="/validation", color="secondary"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Next: Run",
                            href="/run",
                            color="primary",
                            className="float-end",
                        )
                    ),
                ]
            ),
        ]
    )

    yaml_tab = html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Config Path"),
                    dbc.Input(id="train-config-file-path", value="configs/gui_run.yaml"),
                    dbc.Button("Load", id="train-config-load-btn", color="secondary"),
                    dbc.Button("Save", id="train-config-save-btn", color="primary"),
                    dbc.Button("Validate", id="train-config-validate-btn", color="info"),
                    dbc.Button(
                        "Import to Builder", id="train-config-yaml-import-btn", color="warning"
                    ),
                ],
                className="mb-3",
            ),
            dcc.Textarea(
                id="train-config-yaml-preview",
                value=yaml_text,
                className="form-control",
                style={"height": "420px", "fontFamily": "JetBrains Mono, monospace"},
            ),
            html.Pre(id="train-config-validate-result", className="mt-3"),
        ]
    )

    return html.Div(
        [
            html.H2("Train", className="mb-4"),
            guided_mode_banner("train"),
            dbc.Tabs(
                [
                    dbc.Tab(builder, label="Builder", tab_id="train-builder"),
                    dbc.Tab(yaml_tab, label="YAML Source", tab_id="train-yaml"),
                ],
                active_tab="train-builder",
            ),
        ]
    )
