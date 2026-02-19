"""Studio page pane components."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from veldra.gui.components.kpi_cards import kpi_card

_SPLIT_OPTIONS = [
    {"label": "K-Fold", "value": "kfold"},
    {"label": "Stratified", "value": "stratified"},
    {"label": "Group", "value": "group"},
    {"label": "Time Series", "value": "timeseries"},
]

_TASK_OPTIONS = [
    {"label": "Regression", "value": "regression"},
    {"label": "Binary", "value": "binary"},
    {"label": "Multiclass", "value": "multiclass"},
]


def studio_header(mode: str = "train") -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Veldra Studio", className="mb-1"),
                    html.Div("高速モデリング (Phase34.2)", className="small text-muted"),
                ]
            ),
            html.Div(
                [
                    dbc.RadioItems(
                        id="studio-mode-radio",
                        options=[
                            {"label": "学習", "value": "train"},
                            {"label": "推論", "value": "inference"},
                        ],
                        value=mode,
                        inline=True,
                        className="me-3",
                    ),
                    dbc.Button(
                        "モデル管理",
                        id="studio-model-hub-btn",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Guided Mode",
                        id="studio-guided-link",
                        href="/data",
                        color="secondary",
                        outline=True,
                    ),
                ],
                className="d-flex align-items-center",
            ),
        ],
        className="studio-header glass-card mb-3",
    )


def train_scope_pane() -> html.Div:
    return html.Div(
        [
            html.H5("1. Scope", className="mb-3"),
            dcc.Upload(
                id="studio-train-upload",
                children=html.Div(["Drag & Drop or ", html.A("Select Data File")]),
                style={
                    "width": "100%",
                    "height": "96px",
                    "lineHeight": "96px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "10px",
                    "textAlign": "center",
                    "marginBottom": "10px",
                },
                multiple=False,
            ),
            html.Div(id="studio-train-upload-msg", className="small text-muted mb-2"),
            dbc.Label("Target Column", className="fw-bold"),
            dbc.Select(id="studio-train-target-col", options=[]),
            dbc.Label("Task Type", className="fw-bold mt-3"),
            dbc.Select(
                id="studio-train-task-type",
                options=_TASK_OPTIONS,
                value="regression",
            ),
        ],
        className="glass-card h-100",
    )


def train_strategy_pane() -> html.Div:
    return html.Div(
        [
            html.H5("2. Strategy", className="mb-3"),
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            dbc.Label("Split Type", className="fw-bold"),
                            dbc.Select(
                                id="studio-val-split-type",
                                options=_SPLIT_OPTIONS,
                                value="kfold",
                                className="mb-2",
                            ),
                            dbc.Label("N Splits", className="fw-bold"),
                            dbc.Input(
                                id="studio-val-n-splits",
                                type="number",
                                min=2,
                                value=5,
                                className="mb-2",
                            ),
                            dbc.Label("Group Column", className="fw-bold"),
                            dbc.Input(
                                id="studio-val-group-col",
                                placeholder="optional",
                                className="mb-2",
                            ),
                            dbc.Label("Time Column", className="fw-bold"),
                            dbc.Input(
                                id="studio-val-time-col",
                                placeholder="optional",
                                className="mb-2",
                            ),
                            dbc.Label("TimeSeries Mode", className="fw-bold"),
                            dbc.Select(
                                id="studio-val-ts-mode",
                                options=[
                                    {"label": "Expanding", "value": "expanding"},
                                    {"label": "Blocked", "value": "blocked"},
                                ],
                                value="expanding",
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Input(
                                            id="studio-val-test-size",
                                            type="number",
                                            min=1,
                                            placeholder="test_size",
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="studio-val-gap",
                                            type="number",
                                            min=0,
                                            value=0,
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="studio-val-embargo",
                                            type="number",
                                            min=0,
                                            value=0,
                                        ),
                                        width=4,
                                    ),
                                ],
                                className="g-2",
                            ),
                        ],
                        label="Validation",
                    ),
                    dbc.Tab(
                        [
                            dbc.Label("Learning Rate", className="fw-bold"),
                            dcc.Slider(
                                id="studio-model-learning-rate",
                                min=0.001,
                                max=0.3,
                                step=0.001,
                                value=0.05,
                                marks={0.01: "0.01", 0.05: "0.05", 0.1: "0.1", 0.2: "0.2"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Input(
                                            id="studio-model-num-leaves",
                                            type="number",
                                            value=31,
                                            min=2,
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="studio-model-max-depth",
                                            type="number",
                                            value=-1,
                                        ),
                                        width=4,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="studio-model-num-boost-round",
                                            type="number",
                                            value=300,
                                            min=10,
                                        ),
                                        width=4,
                                    ),
                                ],
                                className="g-2 mt-2",
                            ),
                            dbc.Label("Early Stopping Rounds", className="fw-bold mt-3"),
                            dbc.Input(
                                id="studio-model-early-stopping",
                                type="number",
                                min=0,
                                value=100,
                            ),
                        ],
                        label="Model",
                    ),
                    dbc.Tab(
                        [
                            dbc.Switch(
                                id="studio-tune-enabled",
                                label="Enable Optuna tuning",
                                value=False,
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Select(
                                            id="studio-tune-preset",
                                            options=[
                                                {"label": "Fast", "value": "fast"},
                                                {"label": "Standard", "value": "standard"},
                                            ],
                                            value="standard",
                                        ),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="studio-tune-n-trials",
                                            type="number",
                                            min=1,
                                            value=30,
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                            dbc.Label("Objective (optional)", className="fw-bold mt-2"),
                            dbc.Input(id="studio-tune-objective", placeholder="e.g. rmse"),
                        ],
                        label="Tuning",
                    ),
                ]
            ),
        ],
        className="glass-card h-100",
    )


def train_action_pane() -> html.Div:
    return html.Div(
        [
            html.H5("3. Action", className="mb-3"),
            html.Div("READY", id="studio-run-status", className="badge bg-secondary mb-2"),
            dbc.Button(
                [html.I(className="bi bi-play-fill me-2"), "実験を開始 (RUN)"],
                id="studio-run-btn",
                color="primary",
                className="w-100 mb-3",
            ),
            html.Div(id="studio-run-log", className="small text-muted mb-3"),
            html.Div(id="studio-run-progress"),
            html.Div(id="studio-run-kpi", className="mt-3"),
        ],
        className="glass-card h-100",
    )


def inference_scope_pane() -> html.Div:
    return html.Div(
        [
            html.H5("1. Scope (Inference)", className="mb-3"),
            html.Div(
                dbc.Alert("Model is not loaded. Open Model Hub and click Load.", color="warning"),
                id="studio-infer-model-card",
                className="mb-3",
            ),
            dcc.Upload(
                id="studio-predict-upload",
                children=html.Div(["Drag & Drop or ", html.A("Select Inference File")]),
                style={
                    "width": "100%",
                    "height": "96px",
                    "lineHeight": "96px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "10px",
                    "textAlign": "center",
                    "marginBottom": "10px",
                },
                multiple=False,
            ),
            html.Div(id="studio-predict-upload-msg", className="small text-muted mb-2"),
            dbc.Label("Label Column (Optional)", className="fw-bold"),
            dbc.Select(id="studio-predict-label-col", options=[], value=""),
        ],
        className="glass-card h-100",
    )


def inference_spec_pane() -> html.Div:
    return html.Div(
        [
            html.H5("2. Spec", className="mb-3"),
            html.Div(
                "Task / target / metrics are shown after model load.",
                id="studio-infer-spec-summary",
                className="small text-muted mb-2",
            ),
            html.Div(
                "Required feature list will be displayed here.",
                id="studio-infer-feature-list",
                className="small text-muted mb-3",
            ),
            html.Div(
                id="studio-infer-guardrails",
                children=dbc.Alert("No guardrail findings.", color="secondary"),
            ),
        ],
        className="glass-card h-100",
    )


def inference_action_pane() -> html.Div:
    return html.Div(
        [
            html.H5("3. Action", className="mb-3"),
            html.Div("READY", id="studio-predict-status", className="badge bg-secondary mb-2"),
            dbc.Button(
                "予測を開始 (PREDICT)",
                id="studio-predict-btn",
                color="primary",
                className="w-100 mb-2",
            ),
            html.Div(id="studio-predict-log", className="small text-muted mb-2"),
            html.Div(id="studio-predict-progress", className="mb-3"),
            html.Div(id="studio-predict-eval-kpi", className="mb-3"),
            html.Div(id="studio-predict-preview-info", className="small text-muted mb-2"),
            dash_table.DataTable(
                id="studio-predict-preview-table",
                columns=[],
                data=[],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": "12px"},
            ),
            dbc.Button(
                "Download CSV",
                id="studio-predict-download-btn",
                color="secondary",
                className="w-100 mt-2",
            ),
        ],
        className="glass-card h-100",
    )


def model_hub_offcanvas() -> dbc.Offcanvas:
    return dbc.Offcanvas(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Refresh",
                            id="studio-hub-refresh-btn",
                            size="sm",
                            color="secondary",
                        )
                    ),
                    dbc.Col(
                        html.Div(id="studio-hub-page-info", className="small text-muted text-end"),
                        className="d-flex align-items-center justify-content-end",
                    ),
                ],
                className="mb-2",
            ),
            dash_table.DataTable(
                id="studio-hub-table",
                columns=[
                    {"name": "Created", "id": "created_at_utc"},
                    {"name": "Task", "id": "task_type"},
                    {"name": "Run ID", "id": "run_id"},
                    {"name": "Path", "id": "path"},
                ],
                data=[],
                row_selectable="single",
                selected_rows=[],
                page_size=12,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": "12px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Prev",
                            id="studio-hub-prev-btn",
                            size="sm",
                            color="secondary",
                            outline=True,
                        )
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Next",
                            id="studio-hub-next-btn",
                            size="sm",
                            color="secondary",
                            outline=True,
                        )
                    ),
                ],
                className="my-2 g-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Load",
                            id="studio-hub-load-btn",
                            color="primary",
                            className="w-100",
                        )
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Delete",
                            id="studio-hub-delete-btn",
                            color="danger",
                            className="w-100",
                        )
                    ),
                ],
                className="g-2",
            ),
            html.Div(id="studio-hub-feedback", className="small mt-2"),
        ],
        id="studio-model-hub-offcanvas",
        title="Model Hub",
        placement="end",
        is_open=False,
    )


def render_quick_kpis(metrics: dict[str, float]) -> html.Div:
    cards = [kpi_card(label=key, value=f"{value:.6g}") for key, value in list(metrics.items())[:4]]
    if not cards:
        return html.Div("No KPI yet.", className="text-muted small")
    return html.Div(cards, className="d-flex flex-wrap")
