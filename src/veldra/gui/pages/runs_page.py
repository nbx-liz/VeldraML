"""Runs page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from veldra.gui.components.guided_mode_banner import guided_mode_banner


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Runs", className="mb-4"),
            guided_mode_banner("runs"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Select(
                            id="runs-status-filter",
                            options=[
                                {"label": "All Status", "value": ""},
                                {"label": "Queued", "value": "queued"},
                                {"label": "Running", "value": "running"},
                                {"label": "Succeeded", "value": "succeeded"},
                                {"label": "Failed", "value": "failed"},
                                {"label": "Canceled", "value": "canceled"},
                            ],
                            value="",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Select(
                            id="runs-action-filter",
                            options=[
                                {"label": "All Actions", "value": ""},
                                {"label": "Fit", "value": "fit"},
                                {"label": "Tune", "value": "tune"},
                                {"label": "Evaluate", "value": "evaluate"},
                                {"label": "Predict", "value": "predict"},
                                {"label": "Export", "value": "export"},
                                {"label": "Estimate DR", "value": "estimate_dr"},
                            ],
                            value="",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Input(id="runs-search", placeholder="Search run id / artifact"), width=4
                    ),
                    dbc.Col(
                        dbc.Button("Refresh", id="runs-refresh-btn", color="secondary"), width=2
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Select(
                            id="runs-page-size",
                            options=[
                                {"label": "25", "value": 25},
                                {"label": "50", "value": 50},
                                {"label": "100", "value": 100},
                            ],
                            value=50,
                        ),
                        width=2,
                    ),
                    dbc.Col(
                        html.Div(id="runs-page-info", className="small text-muted mt-2"),
                        width=4,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button(
                                    "Prev",
                                    id="runs-page-prev-btn",
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                    className="me-1",
                                ),
                                dbc.Button(
                                    "Next",
                                    id="runs-page-next-btn",
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                ),
                            ]
                        ),
                        width=2,
                    ),
                ],
                className="mb-2 g-2",
            ),
            dash_table.DataTable(
                id="runs-table",
                columns=[
                    {"name": "job_id", "id": "job_id"},
                    {"name": "status", "id": "status"},
                    {"name": "action", "id": "action"},
                    {"name": "created", "id": "created_at_utc"},
                    {"name": "started", "id": "started_at_utc"},
                    {"name": "finished", "id": "finished_at_utc"},
                    {"name": "artifact", "id": "artifact_path"},
                    {"name": "export", "id": "export_shortcut", "presentation": "markdown"},
                    {"name": "re-evaluate", "id": "reeval_shortcut", "presentation": "markdown"},
                ],
                data=[],
                row_selectable="multi",
                page_size=50,
                style_table={"overflowX": "auto"},
            ),
            dcc.Store(id="runs-page", data=0),
            dcc.Store(id="runs-total", data=0),
            dcc.Store(id="runs-selection-store"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Compare Selected",
                            id="runs-compare-btn",
                            color="primary",
                            title="Select exactly two runs to compare metrics and config.",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Clone",
                            id="runs-clone-btn",
                            color="secondary",
                            title="Copy selected run config to Train page.",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Delete",
                            id="runs-delete-btn",
                            color="danger",
                            outline=True,
                            title="Delete selected run records (artifact files remain).",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "View Results",
                            id="runs-view-results-btn",
                            color="success",
                            outline=True,
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Migrate Config", id="runs-migrate-btn", color="warning", outline=True
                        ),
                        width="auto",
                    ),
                ],
                className="my-3 g-2",
            ),
            html.Div(id="runs-feedback", className="mb-2"),
            html.Pre(id="runs-detail", style={"maxHeight": "320px", "overflowY": "auto"}),
        ]
    )
