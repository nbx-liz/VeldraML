"""Reusable help UI components for GUI guidance."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from veldra.gui.components.help_texts import HELP_TEXTS

_SEVERITY_TO_COLOR = {
    "error": "danger",
    "warning": "warning",
    "info": "info",
    "ok": "success",
}

_DEFAULT_ENTRY = {
    "short": "Guidance is not available for this topic yet.",
    "detail": "Use the default recommendation for this field and check validation before run.",
}


def _help_entry(topic_key: str) -> dict[str, str]:
    entry = HELP_TEXTS.get(topic_key)
    if not isinstance(entry, dict):
        return dict(_DEFAULT_ENTRY)
    short = str(entry.get("short") or _DEFAULT_ENTRY["short"])
    detail = str(entry.get("detail") or _DEFAULT_ENTRY["detail"])
    return {"short": short, "detail": detail}


def help_icon(topic_key: str) -> html.Span:
    """Render a compact help icon with tooltip."""
    entry = _help_entry(topic_key)
    icon_id = f"help-icon-{topic_key}"
    return html.Span(
        [
            dbc.Badge("i", id=icon_id, color="secondary", pill=True, className="ms-2"),
            dbc.Tooltip(entry["detail"], target=icon_id, placement="right"),
        ],
        className="d-inline-flex align-items-center",
    )


def context_card(title: str, body: str, variant: str = "info") -> dbc.Card:
    """Render a compact context card."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="fw-bold mb-1"),
                html.Div(body, className="small"),
            ]
        ),
        color=variant,
        outline=variant not in {"info", "warning", "danger", "success", "primary"},
        className="mb-2",
    )


def recommendation_badge(text: str, level: str = "info") -> dbc.Badge:
    """Render recommendation badge."""
    color = _SEVERITY_TO_COLOR.get(level, "secondary")
    return dbc.Badge(text, color=color, className="me-2")


def guide_alert(messages: list[str], severity: str = "info") -> dbc.Alert:
    """Render multi-message guidance alert."""
    color = _SEVERITY_TO_COLOR.get(severity, "secondary")
    body = html.Ul([html.Li(str(msg)) for msg in messages], className="mb-0")
    return dbc.Alert(body, color=color, className="mb-2 py-2 px-3")
