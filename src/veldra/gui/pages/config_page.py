"""Config editor page."""
from __future__ import annotations

from typing import Any

from dash import dcc, html
import dash_bootstrap_components as dbc


def _render_builder_tab() -> html.Div:
    return html.Div(
        [
            html.Div(id="cfg-builder-top"),
            html.Div(
                [
                    html.Div(
                        "Quick Actions",
                        className="small text-uppercase fw-bold text-muted",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Run Now →",
                                id="config-run-now-btn",
                                href="/run",
                                color="success",
                                size="sm",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Jump to Export",
                                id="config-jump-export-btn",
                                href="#cfg-export-section",
                                color="secondary",
                                outline=True,
                                size="sm",
                            ),
                        ],
                    ),
                ],
                className="d-flex justify-content-between align-items-center mb-3 p-2 rounded",
                style={
                    "position": "sticky",
                    "top": "8px",
                    "zIndex": "100",
                    "backgroundColor": "rgba(15, 23, 42, 0.95)",
                    "backdropFilter": "blur(4px)",
                    "border": "1px solid rgba(148, 163, 184, 0.2)",
                },
            ),
            # --- Task Section ---
            html.H5("1. Task Type", className="mb-3 mt-2 text-info"),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.RadioItems(
                            id="cfg-task-type",
                            options=[
                                {"label": "Regression", "value": "regression"},
                                {"label": "Binary Classification", "value": "binary"},
                                {"label": "Multiclass Classification", "value": "multiclass"},
                                {"label": "Frontier (Quantile)", "value": "frontier"},
                            ],
                            value="regression",
                            inline=True,
                            className="mb-2",
                            labelClassName="me-3"
                        ),
                        html.Div(
                            [
                                html.Div([
                                    html.Label("Causal Analysis", className="me-2 fw-bold text-warning"),
                                    dbc.Switch(id="cfg-causal-enabled", value=False, label="Enable DR / DR-DiD"),
                                ], className="d-flex align-items-center mb-2"),
                                
                                html.Div(
                                    [
                                        dbc.Select(
                                            id="cfg-causal-method",
                                            options=[
                                                {"label": "Doubly Robust (DR)", "value": "dr"},
                                                {"label": "DR Diff-in-Diff (DR-DiD)", "value": "dr_did"},
                                            ],
                                            placeholder="Select Method...",
                                        )
                                    ],
                                    id="cfg-causal-method-container",
                                    style={"display": "none"}, # Managed by callback
                                )
                            ],
                            className="mt-2 border-top pt-2"
                        )
                    ]
                ),
                className="mb-4 glass-card p-2"
            ),

            # --- Data Section (Read-only + Column Selectors) ---
            html.H5("2. Data Settings", className="mb-3 text-info"),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Data Path (from Data Page)", className="small text-muted"),
                                        dbc.Input(id="cfg-data-path", disabled=True, className="mb-2"),
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Target Column (Select here)", className="small text-info fw-bold"),
                                        dbc.Select(id="cfg-data-target", placeholder="Select Target...", className="mb-2"),
                                    ],
                                    width=6
                                ),
                            ]
                        ),
                        html.Label("ID Columns (Optional - for Group K-Fold)", className="mt-2"),
                        html.Div(
                            dbc.Checklist(id="cfg-data-id-cols", options=[], value=[], inline=False), 
                            id="cfg-container-id-cols",
                            className="checklist-container mb-2",
                            style={"display": "none"} # Conditional visibility
                        ),
                        
                        html.Label("Categorical Columns (Optional override)", className="mt-2"),
                        html.Div(dbc.Checklist(id="cfg-data-cat-cols", options=[], value=[], inline=False), className="checklist-container mb-2"),
                        
                        html.Label("Exclude Columns (all other non-ID columns used automatically)", className="mt-2 text-info"),
                        html.Div(dbc.Checklist(id="cfg-data-drop-cols", options=[], value=[], inline=False), className="checklist-container mb-2"),
                    ]
                ),
                className="mb-4 glass-card p-2"
            ),

            # --- Split Section ---
            html.H5("3. Split Strategy", className="mb-3 text-info"),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Split Type"),
                                        dbc.Select(
                                            id="cfg-split-type",
                                            options=[
                                                {"label": "K-Fold Cross Validation", "value": "kfold"},
                                                {"label": "Stratified K-Fold", "value": "stratified"},
                                                {"label": "Group K-Fold", "value": "group"},
                                                {"label": "Time Series", "value": "timeseries"},
                                            ],
                                            value="kfold",
                                            className="mb-3"
                                        ),
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Number of Splits"),
                                        dbc.Input(id="cfg-split-nsplits", type="number", value=5, min=2, max=20, step=1),
                                    ],
                                    width=3
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Seed"),
                                        dbc.Input(id="cfg-split-seed", type="number", value=42),
                                    ],
                                    width=3
                                ),
                            ]
                        ),
                        # Conditional Inputs
                        html.Div(
                            [
                                html.Label("Group Column"),
                                dbc.Select(id="cfg-split-group-col", placeholder="Select group column..."),
                            ],
                            id="cfg-container-group",
                            style={"display": "none"},
                            className="mt-2"
                        ),
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Time Column"),
                                                dbc.Select(id="cfg-split-time-col", placeholder="Select time column..."),
                                            ],
                                            width=6
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Mode"),
                                                dbc.Select(
                                                    id="cfg-split-ts-mode", 
                                                    options=[{"label": "Expanding", "value": "expanding"}, {"label": "Blocked", "value": "blocked"}],
                                                    value="expanding"
                                                ),
                                            ],
                                            width=6
                                        ),
                                    ],
                                    className="mb-2"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col([html.Label("Test Size (steps)"), dbc.Input(id="cfg-split-test-size", type="number")], width=4),
                                        dbc.Col([html.Label("Gap"), dbc.Input(id="cfg-split-gap", type="number", value=0)], width=4),
                                        dbc.Col([html.Label("Embargo"), dbc.Input(id="cfg-split-embargo", type="number", value=0)], width=4),
                                    ]
                                )
                            ],
                            id="cfg-container-timeseries",
                            style={"display": "none"},
                            className="mt-2"
                        ),
                    ]
                ),
                className="mb-4 glass-card p-2"
            ),

            # --- Train Section (LightGBM) ---
            html.H5("4. Training Parameters (LightGBM)", className="mb-3 text-info"),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Learning Rate"),
                                        dcc.Slider(
                                            id="cfg-train-lr",
                                            min=0.001, max=0.3, step=0.001,
                                            value=0.1,
                                            marks={0.01: "0.01", 0.05: "0.05", 0.1: "0.1", 0.2: "0.2"},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ],
                                    width=6,
                                    className="pe-3"
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Num Leaves"),
                                        dbc.Input(id="cfg-train-num-leaves", type="number", value=31),
                                    ],
                                    width=3
                                ),
                                dbc.Col(
                                    [
                                        html.Label("N Estimators"),
                                        dbc.Input(id="cfg-train-n-estimators", type="number", value=100),
                                    ],
                                    width=3
                                ),
                            ],
                            className="mb-4"
                        ),
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col([html.Label("Max Depth"), dbc.Input(id="cfg-train-max-depth", type="number", value=-1)], width=4),
                                                dbc.Col([html.Label("Min Child Samples"), dbc.Input(id="cfg-train-min-child", type="number", value=20)], width=4),
                                                dbc.Col([html.Label("Early Stopping"), dbc.Input(id="cfg-train-early-stop", type="number", value=100)], width=4),
                                            ],
                                            className="mb-3"
                                        ),
                                        html.Label("Subsample"),
                                        dcc.Slider(id="cfg-train-subsample", min=0.1, max=1.0, step=0.1, value=1.0, marks={0.5: "0.5", 1.0: "1.0"}),
                                        html.Label("Colsample By Tree", className="mt-3"),
                                        dcc.Slider(id="cfg-train-colsample", min=0.1, max=1.0, step=0.1, value=1.0, marks={0.5: "0.5", 1.0: "1.0"}),
                                        
                                        dbc.Row(
                                            [
                                                dbc.Col([html.Label("Reg Alpha"), dbc.Input(id="cfg-train-reg-alpha", type="number", value=0)], width=6),
                                                dbc.Col([html.Label("Reg Lambda"), dbc.Input(id="cfg-train-reg-lambda", type="number", value=0)], width=6),
                                            ],
                                            className="mt-3"
                                        )
                                    ],
                                    title="Advanced Training Parameters"
                                )
                            ],
                            start_collapsed=True
                        )
                    ]
                ),
                className="mb-4 glass-card p-2"
            ),

            # --- Tuning Section ---
            html.H5("5. Hyperparameter Tuning (Optuna)", className="mb-3 text-info"),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Switch(id="cfg-tune-enabled", label="Enable Tuning", value=False, className="fs-5"),
                                    width=4
                                ),
                            ],
                            className="mb-3"
                        ),
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Preset"),
                                                dbc.RadioItems(
                                                    id="cfg-tune-preset",
                                                    options=[
                                                        {"label": "Fast (Light)", "value": "fast"},
                                                        {"label": "Standard", "value": "standard"},
                                                    ],
                                                    value="standard",
                                                    inline=True
                                                )
                                            ],
                                            width=6
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("N Trials"),
                                                dbc.Input(id="cfg-tune-trials", type="number", value=30),
                                            ],
                                            width=3
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Objective"),
                                                dbc.Select(id="cfg-tune-objective", placeholder="Auto"),
                                            ],
                                            width=3
                                        ),
                                    ]
                                )
                            ],
                            id="cfg-container-tune",
                            style={"display": "none"}
                        ),
                        # Search Space Configuration
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        html.P("Override default search ranges for Optuna.", className="text-muted small"),
                                        dbc.Row([
                                            dbc.Col([html.Label("LR Min"), dbc.Input(id="cfg-tune-lr-min", placeholder="1e-4", type="number")], width=3),
                                            dbc.Col([html.Label("LR Max"), dbc.Input(id="cfg-tune-lr-max", placeholder="0.3", type="number")], width=3),
                                            dbc.Col([html.Label("Leaves Min"), dbc.Input(id="cfg-tune-leaves-min", placeholder="16", type="number")], width=3),
                                            dbc.Col([html.Label("Leaves Max"), dbc.Input(id="cfg-tune-leaves-max", placeholder="255", type="number")], width=3),
                                        ], className="mb-2"),
                                        dbc.Row([
                                            dbc.Col([html.Label("Depth Min"), dbc.Input(id="cfg-tune-depth-min", placeholder="-1", type="number")], width=3),
                                            dbc.Col([html.Label("Depth Max"), dbc.Input(id="cfg-tune-depth-max", placeholder="15", type="number")], width=3),
                                            dbc.Col([html.Label("Feat Frac Min"), dbc.Input(id="cfg-tune-ff-min", placeholder="0.1", type="number")], width=3),
                                            dbc.Col([html.Label("Feat Frac Max"), dbc.Input(id="cfg-tune-ff-max", placeholder="1.0", type="number")], width=3),
                                        ]),
                                    ],
                                    title="Advanced: Custom Search Space"
                                )
                            ],
                            start_collapsed=True,
                            className="mt-3"
                        )
                    ]
                ),
                className="mb-4 glass-card p-2"
            ),
            
            # --- Export Section ---
             html.Div(id="cfg-export-section"),
             html.H5("6. Export", className="mb-3 text-info"),
             dbc.Card(
                 dbc.CardBody(
                     [
                         dbc.Row(
                             [
                                 dbc.Col(
                                     [
                                         html.Label("Artifact Directory"),
                                         dbc.InputGroup(
                                             [
                                                 dbc.Select(
                                                     id="cfg-export-dir-preset",
                                                     options=[
                                                         {"label": "Standard (artifacts/)", "value": "artifacts"},
                                                         {"label": "Output (output/)", "value": "output"},
                                                         {"label": "Custom...", "value": "custom"},
                                                     ],
                                                     placeholder="Presets...",
                                                     style={"maxWidth": "150px"}
                                                 ),
                                                 dbc.Input(id="cfg-export-dir", value="artifacts"),
                                             ]
                                         ),
                                     ],
                                     width=6
                                 )
                             ]
                         )
                     ]
                 ),
                 className="mb-4 glass-card p-2"
             ),
             
             # --- Actions ---
             html.Div(
                 [
                     dbc.Button("Next: Run →", id="config-to-run-btn", color="success", size="lg", href="/run", className="w-100 shadow mb-2"),
                     dbc.Button("Back to Top", id="config-back-top-btn", color="secondary", outline=True, href="#cfg-builder-top", className="w-100"),
                 ],
                 className="mt-4 mb-5"
             )
        ]
    )


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Configuration Builder", className="mb-4"),
            
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Builder", 
                        tab_id="tab-builder",
                        children=[
                            html.Div(
                                _render_builder_tab(),
                                className="p-3"
                            )
                        ]
                    ),
                    dbc.Tab(
                        label="YAML Source", 
                        tab_id="tab-yaml",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Config Path"),
                                                    dbc.Input(
                                                        id="config-file-path",
                                                        type="text",
                                                        value="configs/gui_run.yaml",
                                                    ),
                                                    dbc.Button("Load", id="config-load-btn", color="secondary"),
                                                    dbc.Button("Save", id="config-save-btn", color="primary"),
                                                ],
                                                className="mb-3",
                                            ),
                                            html.Label("Generated RunConfig YAML", className="fw-bold mb-2"),
                                            dcc.Textarea(
                                                id="config-yaml",
                                                className="form-control mb-3",
                                                style={
                                                    "width": "100%", 
                                                    "height": "500px", 
                                                    "fontFamily": "JetBrains Mono, monospace",
                                                    "backgroundColor": "#0d0d0d",
                                                    "color": "#e2e8f0",
                                                    "border": "1px solid rgba(148, 163, 184, 0.1)"
                                                },
                                            ),
                                            dbc.Button("Validate Configuration", id="config-validate-btn", color="info", className="mb-3"),
                                            dbc.Button("Import from Builder", id="config-import-btn", color="warning", outline=True, className="float-end"),
                                            html.Pre(
                                                id="config-validation-result",
                                                className="p-3",
                                                style={"display": "none"},
                                            ),
                                        ],
                                        className="glass-card p-4 mt-3"
                                    )
                                ]
                            )
                        ]
                    ),
                    dbc.Tab(
                        label="Migration", 
                        tab_id="tab-migration", 
                        children=[
                            html.Div(
                                [
                                    html.H4("Config Migration Tool", className="mb-3"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Input Path"),
                                                    dbc.Input(id="config-migrate-input-path", value="configs/run.yaml", className="mb-3"),
                                                ],
                                                width=6
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Target Version"),
                                                    dbc.Input(id="config-migrate-target-version", type="number", value=1, min=1, className="mb-3"),
                                                ],
                                                width=6
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button("Preview", id="config-migrate-preview-btn", color="info", className="me-2"),
                                            dbc.Button("Apply", id="config-migrate-apply-btn", color="warning"),
                                        ],
                                        className="mb-4"
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label("Normalized Preview"),
                                                    dcc.Textarea(
                                                        id="config-migrate-normalized-yaml",
                                                        className="form-control",
                                                        style={"height": "300px", "fontFamily": "JetBrains Mono"},
                                                    )
                                                ],
                                                width=6
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label("Diff"),
                                                    html.Pre(
                                                        id="config-migrate-diff",
                                                        style={"height": "300px", "overflowY": "auto", "backgroundColor": "#0d0d0d"}
                                                    )
                                                ],
                                                width=6
                                            ),
                                        ]
                                    ),
                                    html.Div(id="config-migrate-result", className="mt-3")
                                ],
                                className="glass-card p-4 mt-3"
                            )
                        ]
                    ),
                ],
                className="nav-fill",
                active_tab="tab-builder",
            ),
        ]
    )
