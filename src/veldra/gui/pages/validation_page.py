"""Validation settings page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from veldra.gui.components.help_ui import help_icon


def layout(state: dict | None = None) -> html.Div:
    state = state or {}
    split = state.get("split_config") or {}
    return html.Div(
        [
            html.H2("Validation", className="mb-4"),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(id="validation-split-context", className="mb-2"),
                        html.Div(id="validation-recommendation", className="mb-2"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Split Type", className="fw-bold"),
                                        dbc.Select(
                                            id="validation-split-type",
                                            options=[
                                                {"label": "K-Fold", "value": "kfold"},
                                                {"label": "Stratified", "value": "stratified"},
                                                {"label": "Group", "value": "group"},
                                                {"label": "Time Series", "value": "timeseries"},
                                            ],
                                            value=split.get("type", "kfold"),
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("N Splits", className="fw-bold"),
                                        dbc.Input(
                                            id="validation-n-splits",
                                            type="number",
                                            value=split.get("n_splits", 5),
                                            min=2,
                                            max=20,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Seed", className="fw-bold"),
                                        dbc.Input(
                                            id="validation-seed",
                                            type="number",
                                            value=split.get("seed", 42),
                                        ),
                                    ],
                                    width=4,
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                html.Label("Group Column"),
                                dbc.Select(
                                    id="validation-group-col",
                                    value=split.get("group_col"),
                                ),
                            ],
                            id="validation-group-container",
                            style={"display": "none"},
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Time Column"),
                                                dbc.Select(
                                                    id="validation-time-col",
                                                    value=split.get("time_col"),
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Mode", className="mb-0"),
                                                        help_icon("timeseries_mode"),
                                                    ],
                                                    className=(
                                                        "d-flex align-items-center gap-1 mb-1"
                                                    ),
                                                ),
                                                dbc.Select(
                                                    id="validation-ts-mode",
                                                    options=[
                                                        {
                                                            "label": "Expanding",
                                                            "value": "expanding",
                                                        },
                                                        {"label": "Blocked", "value": "blocked"},
                                                    ],
                                                    value=split.get("timeseries_mode", "expanding"),
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Test Size"),
                                                dbc.Input(
                                                    id="validation-test-size",
                                                    type="number",
                                                    value=split.get("test_size"),
                                                ),
                                            ],
                                            width=4,
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Gap"),
                                                html.Div(
                                                    help_icon("timeseries_gap"), className="mb-1"
                                                ),
                                                dbc.Input(
                                                    id="validation-gap",
                                                    type="number",
                                                    value=split.get("gap", 0),
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Embargo"),
                                                html.Div(
                                                    help_icon("timeseries_embargo"),
                                                    className="mb-1",
                                                ),
                                                dbc.Input(
                                                    id="validation-embargo",
                                                    type="number",
                                                    value=split.get("embargo", 0),
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                            ],
                            id="validation-timeseries-container",
                            style={"display": "none"},
                            className="mb-3",
                        ),
                        html.Div(id="validation-guardrail-container", className="mb-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button("Back: Target", href="/target", color="secondary"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Next: Train",
                                        href="/train",
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
