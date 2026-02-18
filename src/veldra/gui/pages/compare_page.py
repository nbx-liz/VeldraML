"""Compare page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Compare", className="mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Artifacts (max 5)"),
                            dcc.Dropdown(
                                id="compare-artifacts",
                                options=[],
                                multi=True,
                                placeholder="Select artifacts...",
                            ),
                        ],
                        width=9,
                    ),
                    dbc.Col(
                        [
                            html.Label("Baseline"),
                            dbc.Select(id="compare-baseline", options=[]),
                        ],
                        width=3,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="compare-checks", className="mb-3"),
            dash_table.DataTable(
                id="compare-metrics-table",
                columns=[
                    {"name": "metric", "id": "metric"},
                    {"name": "artifact", "id": "artifact"},
                    {"name": "value", "id": "value"},
                    {"name": "baseline", "id": "baseline"},
                    {"name": "delta_from_baseline", "id": "delta_from_baseline"},
                ],
                data=[],
                page_size=12,
                style_table={"overflowX": "auto"},
            ),
            dcc.Graph(id="compare-chart", style={"height": "360px"}, className="mb-3"),
            html.H5("Config Diff"),
            html.Div(id="compare-config-diff"),
            dbc.Button(
                "Export Comparison Report",
                id="compare-export-btn",
                color="secondary",
                className="mt-3",
            ),
            html.Div(id="compare-feedback", className="mt-2"),
        ]
    )
