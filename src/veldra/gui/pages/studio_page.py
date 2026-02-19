"""Studio page layout."""

from __future__ import annotations

from dash import dcc, html

from veldra.gui.components import studio_parts


def layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="store-studio-mode", data="train"),
            dcc.Store(id="store-studio-train-data", data={}),
            dcc.Store(id="store-studio-predict-data", data={}),
            dcc.Store(id="store-studio-artifact", data={}),
            dcc.Store(id="store-studio-last-job", data={}),
            dcc.Store(id="store-studio-predict-job", data={}),
            dcc.Store(id="store-studio-hub-page", data=0),
            dcc.Store(id="store-studio-hub-total", data=0),
            dcc.Store(id="store-studio-predict-result", data={}),
            dcc.Interval(
                id="studio-run-poll-interval",
                interval=2000,
                n_intervals=0,
                disabled=True,
            ),
            dcc.Interval(
                id="studio-predict-poll-interval",
                interval=2000,
                n_intervals=0,
                disabled=True,
            ),
            dcc.Download(id="studio-predict-csv-download"),
            dcc.ConfirmDialog(id="studio-hub-delete-confirm", message="Delete selected artifact?"),
            studio_parts.studio_header("train"),
            studio_parts.model_hub_offcanvas(),
            html.Div(
                [
                    html.Div(studio_parts.train_scope_pane(), id="studio-pane-scope"),
                    html.Div(studio_parts.train_strategy_pane(), id="studio-pane-center"),
                    html.Div(studio_parts.train_action_pane(), id="studio-pane-action"),
                ],
                className="studio-grid",
            ),
        ],
        id="studio-root",
    )
