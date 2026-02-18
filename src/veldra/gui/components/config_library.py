"""Config template/custom library controls."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def render_config_library(
    prefix: str, *, template_options: list[dict], slot_options: list[dict]
) -> html.Div:
    return html.Div(
        [
            html.H5("Config Library", className="text-info mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Built-in Template"),
                            dbc.Select(
                                id=f"{prefix}-template-select",
                                options=template_options,
                                value=(template_options[0]["value"] if template_options else None),
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Custom Config Slots"),
                            dbc.Select(
                                id=f"{prefix}-slot-select",
                                options=slot_options,
                                value=(slot_options[0]["value"] if slot_options else None),
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id=f"{prefix}-slot-name", placeholder="Slot name", value="Custom Config"
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Apply Template",
                                    id=f"{prefix}-template-apply-btn",
                                    color="primary",
                                    size="sm",
                                ),
                                dbc.Button(
                                    "Save Slot",
                                    id=f"{prefix}-slot-save-btn",
                                    color="secondary",
                                    size="sm",
                                ),
                                dbc.Button(
                                    "Load Slot",
                                    id=f"{prefix}-slot-load-btn",
                                    color="secondary",
                                    size="sm",
                                ),
                                dbc.Button(
                                    "Clone Slot",
                                    id=f"{prefix}-slot-clone-btn",
                                    color="warning",
                                    size="sm",
                                ),
                                dbc.Button(
                                    "Open Wizard",
                                    id=f"{prefix}-wizard-open-btn",
                                    color="info",
                                    size="sm",
                                ),
                            ]
                        ),
                        width=8,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Diff Base"),
                            dbc.RadioItems(
                                id=f"{prefix}-diff-base",
                                options=[
                                    {"label": "Template", "value": "template"},
                                    {"label": "Selected Slot", "value": "slot"},
                                ],
                                value="template",
                                inline=True,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(id=f"{prefix}-diff-count", className="small text-muted mt-4"),
                        width=6,
                    ),
                ],
                className="mb-2",
            ),
            html.Pre(
                id=f"{prefix}-diff-view",
                className="p-2 small",
                style={"maxHeight": "180px", "overflow": "auto", "backgroundColor": "#0d0d0d"},
            ),
            html.Div(id=f"{prefix}-library-message", className="small mt-2"),
            dcc.Store(id=f"{prefix}-wizard-step", data=1),
        ],
        className="glass-card p-3 mb-3",
    )
