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
                            html.Label("Run A"),
                            dbc.Select(id="compare-artifact-a", options=[]),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Run B"),
                            dbc.Select(id="compare-artifact-b", options=[]),
                        ],
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="compare-checks", className="mb-3"),
            dash_table.DataTable(
                id="compare-metrics-table",
                columns=[
                    {"name": "metric", "id": "metric"},
                    {"name": "run_a", "id": "run_a"},
                    {"name": "run_b", "id": "run_b"},
                    {"name": "delta", "id": "delta"},
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
