"""Guardrail rendering helpers."""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html

_LEVEL_TO_COLOR = {
    "error": "danger",
    "warning": "warning",
    "info": "info",
    "ok": "success",
}


def render_guardrails(items: list[dict[str, Any]] | None) -> html.Div:
    """Render guardrail result list as alerts."""
    alerts: list[Any] = []
    for item in items or []:
        level = str(item.get("level", "info")).lower()
        message = str(item.get("message", ""))
        suggestion = item.get("suggestion")
        color = _LEVEL_TO_COLOR.get(level, "secondary")
        body = [html.Div(message)]
        if suggestion:
            body.append(html.Div(f"Suggestion: {suggestion}", className="small mt-1"))
        alerts.append(
            dbc.Alert(body, color=color, className="mb-2 py-2 px-3")
        )

    if not alerts:
        alerts.append(dbc.Alert("No guardrail findings.", color="secondary", className="mb-0"))

    return html.Div(alerts, className="guardrail-panel")
