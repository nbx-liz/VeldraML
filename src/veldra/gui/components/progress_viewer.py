"""Progress and log viewer for run jobs."""

from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import html


def render_progress_viewer(
    *,
    progress_pct: float,
    current_step: str | None,
    logs: list[dict[str, Any]],
    log_limit: int,
    log_total: int,
) -> html.Div:
    safe_progress = max(0.0, min(float(progress_pct), 100.0))
    step_text = str(current_step or "n/a")
    log_lines = [
        html.Div(
            [
                html.Span(str(item.get("created_at_utc") or "n/a"), className="text-muted me-2"),
                html.Span(
                    str(item.get("level") or "INFO"),
                    className=f"badge bg-{_badge_color(str(item.get('level') or 'INFO'))} me-2",
                ),
                html.Span(str(item.get("message") or "")),
            ],
            className="mb-1",
        )
        for item in logs
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Progress", className="fw-bold me-2"),
                    html.Span(f"{safe_progress:.1f}%"),
                    html.Span(f" | {step_text}", className="text-muted ms-2"),
                ],
                className="mb-2",
            ),
            dbc.Progress(value=safe_progress, striped=True, animated=safe_progress < 100.0),
            html.Div(
                f"Showing {len(logs)} / {log_total} logs (limit={log_limit})",
                className="small text-muted mt-2",
            ),
            html.Div(
                log_lines or [html.Div("No logs yet.", className="text-muted small")],
                style={
                    "maxHeight": "260px",
                    "overflowY": "auto",
                    "border": "1px solid rgba(148, 163, 184, 0.1)",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "backgroundColor": "#0d0d0d",
                },
                className="mt-2",
            ),
        ]
    )


def _badge_color(level: str) -> str:
    key = level.strip().upper()
    if key == "ERROR":
        return "danger"
    if key == "WARNING":
        return "warning"
    if key == "DEBUG":
        return "secondary"
    return "info"
