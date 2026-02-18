"""Config summary card helper."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def render_config_summary(state: dict | None) -> html.Div:
    state = state or {}
    task_type = state.get("task_type") or "regression"
    split = state.get("split_config") or {}
    train = state.get("train_config") or {}
    tuning = state.get("tuning_config") or {}

    rows = [
        html.Li(f"Task: {task_type}"),
        html.Li(
            "Split: "
            f"{split.get('type', 'kfold')} ({split.get('n_splits', 5)} folds)"
        ),
        html.Li(
            "Train: "
            f"lr={train.get('learning_rate', 0.05)}, "
            f"rounds={train.get('num_boost_round', 300)}, "
            f"leaves={train.get('num_leaves', 31)}"
        ),
        html.Li(f"Tuning: {'ON' if tuning.get('enabled') else 'OFF'}"),
    ]

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Setting Summary", className="fw-bold mb-2"),
                html.Ul(rows, className="mb-0"),
            ]
        ),
        className="glass-card",
    )
