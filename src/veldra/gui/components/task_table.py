"""Task Table component."""

from __future__ import annotations

from typing import Any

from dash import dash_table, html


def task_table(id_prefix: str, data: list[dict[str, Any]]) -> html.Div:
    """Create a styled data table for tasks."""

    return html.Div(
        [
            dash_table.DataTable(
                id=f"{id_prefix}-table",
                columns=[
                    {"name": "Status", "id": "status", "presentation": "markdown"},
                    {"name": "Action", "id": "action"},
                    {"name": "Priority", "id": "priority"},
                    {"name": "Created", "id": "created_at_utc"},
                    {"name": "Job ID", "id": "job_id"},
                ],
                data=data,
                style_as_list_view=True,
                style_cell={
                    "padding": "12px 16px",
                    "textAlign": "left",
                    "backgroundColor": "#1a1b2e",
                    "color": "#e2e8f0",
                    "borderBottom": "1px solid rgba(148, 163, 184, 0.1)",
                    "fontFamily": "Inter, sans-serif",
                },
                style_header={
                    "backgroundColor": "#1a1b2e",
                    "fontWeight": "bold",
                    "color": "#94a3b8",
                    "borderBottom": "2px solid rgba(148, 163, 184, 0.1)",
                    "textTransform": "uppercase",
                    "fontSize": "12px",
                    "letterSpacing": "0.05em",
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "rgba(255, 255, 255, 0.02)",
                    },
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "rgba(124, 58, 237, 0.1) !important",
                        "border": "1px solid #7c3aed !important",
                    },
                ],
                row_selectable="single",
                selected_rows=[],
                page_size=10,
            )
        ],
        className="glass-card",
        style={"padding": "0", "overflow": "hidden"},
    )
