"""Data selection page."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
from veldra.gui.components.kpi_cards import kpi_card


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Data Selection", className="mb-4"),
            
            # File Selection Area
            html.Div(
                [
                    dcc.Upload(
                        id="data-upload-drag",
                        children=html.Div(
                            ["Drag & Drop or ", html.A("Select Data File")]
                        ),
                        style={
                            "width": "100%",
                            "height": "100px",
                            "lineHeight": "100px",
                            "borderWidth": "2px",
                            "borderStyle": "dashed",
                            "borderRadius": "10px",
                            "textAlign": "center",
                            "marginBottom": "10px",
                            "borderColor": "var(--text-secondary)",
                            "color": "var(--text-primary)",
                            "cursor": "pointer",
                        },
                        multiple=False
                    ),
                    html.Div(id="data-upload-msg", className="text-center text-info small mb-2"),
                    # Selected file indicator (read-only)
                    html.Div(
                        [
                            html.I(className="bi bi-file-earmark-text me-2"),
                            html.Span("No file selected — upload or drop a file above", id="data-selected-file-label",
                                       className="text-light"),
                        ],
                        id="data-selected-file",
                        className="d-flex align-items-center p-3 mb-3 rounded",
                        style={
                            "backgroundColor": "rgba(124, 58, 237, 0.1)",
                            "border": "1px solid rgba(124, 58, 237, 0.3)",
                        },
                    ),
                    # Hidden store for the actual data path
                    dcc.Store(id="data-file-path", data=""),
                    html.Div(
                        "Data inspection starts automatically after file selection.",
                        className="small text-muted",
                    ),
                    html.Div(id="data-error-message", className="text-danger small mb-3 mt-2"),
                ],
                className="glass-card mb-4",
            ),
            
            # Loading Spinner
            dcc.Loading(
                id="data-loading",
                type="dot",
                children=[html.Div(id="data-inspection-result")],
            ),
        ]
    )

def render_data_stats(stats: dict) -> html.Div:
    """Render data statistics cards."""
    return html.Div(
        [
            html.H4("Dataset Summary", className="mb-3"),
            html.Div(
                [
                    kpi_card("Rows", stats["n_rows"]),
                    kpi_card("Columns", stats["n_cols"]),
                    kpi_card("Numeric Vars", len(stats["numeric_cols"])),
                    kpi_card("Categorical Vars", len(stats["categorical_cols"])),
                    kpi_card("Missing Values", stats["missing_count"], 
                             trend="High" if stats["missing_count"] > 0 else "None", 
                             trend_direction="down" if stats["missing_count"] > 0 else "neutral"),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginBottom": "24px"}
            ),

            # Target selection moved to Config Page
            # Target selection moved to Config Page
            dbc.Button("Next: Configure →", id="data-to-config-btn", color="success", href="/config", className="w-100"),
        ]
    )

def render_data_preview(preview: list[dict]) -> html.Div:
    """Render data preview table."""
    if not preview:
        return html.Div()

    columns = list(preview[0].keys())
    header = html.Thead(
        html.Tr(
            [html.Th(col, style={"whiteSpace": "nowrap"}) for col in columns]
        )
    )
    body_rows = []
    for row in preview:
        body_rows.append(
            html.Tr(
                [
                    html.Td(
                        str(row.get(col, "")),
                        style={
                            "whiteSpace": "nowrap",
                            "verticalAlign": "top",
                        },
                    )
                    for col in columns
                ]
            )
        )
    body = html.Tbody(body_rows)

    return html.Div(
        [
            html.H4("Data Preview (First 10 Rows)", className="mb-3 mt-4"),
            html.Div(
                [
                    html.Div(
                        html.Table(
                            [header, body],
                            className="table table-dark table-striped table-hover mb-0",
                            style={
                                "minWidth": "100%",
                                "tableLayout": "auto",
                            },
                        ),
                        className="data-preview-card",
                        style={
                            "overflowX": "auto",
                            "overflowY": "auto",
                            "maxHeight": "420px",
                        },
                    ),
                ],
                className="glass-card",
            ),
            html.Div(
                "Wide tables are confined to this preview area with horizontal/vertical scroll.",
                className="small text-muted mt-2",
            ),
        ]
    )
