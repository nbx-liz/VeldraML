"""KPI Card component."""

from __future__ import annotations

from typing import Literal

from dash import html


def kpi_card(
    label: str,
    value: str | float | int,
    trend: str | None = None,
    trend_direction: Literal["up", "down", "neutral"] = "neutral",
) -> html.Div:
    """Create a KPI card with label, value, and optional trend."""

    trend_elem = None
    if trend:
        trend_class = "kpi-trend-neutral"
        if trend_direction == "up":
            trend_class = "kpi-trend-up"
        elif trend_direction == "down":
            trend_class = "kpi-trend-down"

        icon = "▲" if trend_direction == "up" else "▼" if trend_direction == "down" else "•"
        trend_elem = html.Span(
            f"{icon} {trend}", className=trend_class, style={"marginLeft": "8px"}
        )

    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(
                [html.Span(str(value), className="kpi-value"), trend_elem],
                style={"display": "flex", "alignItems": "baseline"},
            ),
        ],
        className="glass-card kpi-card",
        style={"minWidth": "200px", "margin": "10px", "flex": "1"},
    )
