"""Quick-start config wizard component."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def render_config_wizard(prefix: str) -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Quick Start Wizard")),
            dbc.ModalBody(
                [
                    html.Div(id=f"{prefix}-wizard-step-label", className="small text-info mb-2"),
                    html.Label("Task Type"),
                    dbc.Select(
                        id=f"{prefix}-wizard-task",
                        options=[
                            {"label": "Regression", "value": "regression"},
                            {"label": "Binary", "value": "binary"},
                            {"label": "Multiclass", "value": "multiclass"},
                            {"label": "Frontier", "value": "frontier"},
                        ],
                        value="regression",
                        className="mb-2",
                    ),
                    html.Label("Data Path"),
                    dbc.Input(id=f"{prefix}-wizard-data-path", className="mb-2"),
                    html.Label("Target Column"),
                    dbc.Input(id=f"{prefix}-wizard-target", className="mb-2"),
                    html.Label("Split Type"),
                    dbc.Select(
                        id=f"{prefix}-wizard-split-type",
                        options=[
                            {"label": "KFold", "value": "kfold"},
                            {"label": "Stratified", "value": "stratified"},
                            {"label": "Group", "value": "group"},
                            {"label": "TimeSeries", "value": "timeseries"},
                        ],
                        value="kfold",
                        className="mb-2",
                    ),
                    html.Label("Learning Rate"),
                    dcc.Slider(
                        id=f"{prefix}-wizard-lr",
                        min=0.001,
                        max=0.3,
                        step=0.001,
                        value=0.05,
                        marks={0.01: "0.01", 0.05: "0.05", 0.1: "0.1"},
                    ),
                    html.Label("Num Boost Round", className="mt-2"),
                    dbc.Input(id=f"{prefix}-wizard-rounds", type="number", value=300),
                    html.Div(id=f"{prefix}-wizard-message", className="small mt-2"),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Back", id=f"{prefix}-wizard-prev-btn", color="secondary", size="sm"
                    ),
                    dbc.Button(
                        "Next", id=f"{prefix}-wizard-next-btn", color="secondary", size="sm"
                    ),
                    dbc.Button(
                        "Apply", id=f"{prefix}-wizard-apply-btn", color="primary", size="sm"
                    ),
                    dbc.Button("Close", id=f"{prefix}-wizard-close-btn", color="link", size="sm"),
                ]
            ),
        ],
        id=f"{prefix}-wizard-modal",
        is_open=False,
        size="lg",
    )
