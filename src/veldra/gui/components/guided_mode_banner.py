"""Reusable Guided Mode banner for classic multi-page flow."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def _normalize_page_key(page_key: str) -> str:
    key = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in page_key).strip("-")
    return key or "page"


def guided_mode_banner(page_key: str, studio_href: str = "/studio") -> dbc.Alert:
    """Render Guided Mode banner with a direct Studio shortcut."""
    safe_key = _normalize_page_key(page_key)
    return dbc.Alert(
        [
            html.Span(
                "This page is Guided Mode. For faster experimentation, open Studio.",
                className="me-2",
            ),
            dbc.Button(
                "Open Studio",
                id=f"guided-mode-open-studio-{safe_key}",
                href=studio_href,
                color="info",
                outline=True,
                size="sm",
                className="ms-auto",
            ),
        ],
        id=f"guided-mode-banner-{safe_key}",
        color="info",
        className="guided-mode-banner d-flex align-items-center gap-2 mb-3",
    )
