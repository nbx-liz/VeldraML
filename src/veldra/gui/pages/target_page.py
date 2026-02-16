"""Target settings page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def layout(state: dict | None = None) -> html.Div:
    state = state or {}
    return html.Div(
        [
            html.H2("Target", className="mb-4"),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Data Path", className="fw-bold"),
                                        dbc.Input(
                                            id="target-data-path",
                                            value=state.get("data_path", ""),
                                            disabled=True,
                                        ),
                                    ],
                                    width=12,
                                    className="mb-3",
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Target Column", className="fw-bold"),
                                        dbc.Select(
                                            id="target-col-select",
                                            placeholder="Select target column...",
                                        ),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Task Type", className="fw-bold"),
                                        dbc.RadioItems(
                                            id="target-task-type",
                                            options=[
                                                {"label": "Regression", "value": "regression"},
                                                {"label": "Binary", "value": "binary"},
                                                {"label": "Multiclass", "value": "multiclass"},
                                                {"label": "Frontier", "value": "frontier"},
                                            ],
                                            value=state.get("task_type", "regression"),
                                            inline=True,
                                        ),
                                        html.Div(
                                            id="target-task-hint",
                                            className="small text-info mt-1",
                                        ),
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        dbc.Switch(
                                            id="target-causal-enabled",
                                            label="Enable causal mode",
                                            value=bool(
                                                (state.get("causal_config") or {}).get("enabled")
                                            ),
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Select(
                                                        id="target-causal-method",
                                                        options=[
                                                            {"label": "DR", "value": "dr"},
                                                            {"label": "DR-DiD", "value": "dr_did"},
                                                        ],
                                                        placeholder="Causal method",
                                                        value=(
                                                            state.get("causal_config") or {}
                                                        ).get("method"),
                                                    ),
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Select(
                                                        id="target-treatment-col",
                                                        placeholder="Treatment column",
                                                        value=(
                                                            state.get("causal_config") or {}
                                                        ).get("treatment_col"),
                                                    ),
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    dbc.Select(
                                                        id="target-unit-id-col",
                                                        placeholder="Unit ID column",
                                                        value=(
                                                            state.get("causal_config") or {}
                                                        ).get("unit_id_col"),
                                                    ),
                                                    width=4,
                                                ),
                                            ],
                                            className="mt-2",
                                        ),
                                    ],
                                    title="Causal Settings",
                                ),
                                dbc.AccordionItem(
                                    [
                                        html.Label("Exclude Columns"),
                                        dbc.Checklist(
                                            id="target-exclude-cols",
                                            options=[],
                                            value=state.get("exclude_cols", []),
                                            inline=False,
                                        ),
                                    ],
                                    title="Excluded Columns",
                                ),
                            ],
                            start_collapsed=True,
                            className="mb-3",
                        ),
                        html.Div(id="target-guardrail-container", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button("Back: Data", href="/data", color="secondary"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Next: Validation",
                                        href="/validation",
                                        color="primary",
                                        className="float-end",
                                    ),
                                ),
                            ]
                        ),
                    ]
                ),
                className="glass-card",
            ),
        ]
    )
