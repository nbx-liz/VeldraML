"""Data selection page."""
from __future__ import annotations

from dash import dcc, html, dash_table
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
                                       className="text-muted"),
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
                    dbc.Button("Inspect Data", id="data-inspect-btn", color="primary", className="w-100"),
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
        
    cols = [{"name": i, "id": i} for i in preview[0].keys()]
    
    return html.Div(
        [
            html.H4("Data Preview (First 10 Rows)", className="mb-3 mt-4"),
            html.Div(
                dash_table.DataTable(
                    columns=cols,
                    data=preview,
                    style_as_list_view=True,
                    style_cell={
                        "padding": "10px",
                        "textAlign": "left",
                        "backgroundColor": "#1a1b2e",
                        "color": "#e2e8f0",
                        "borderBottom": "1px solid rgba(148, 163, 184, 0.1)",
                        "minWidth": "100px",
                    },
                    style_header={
                        "backgroundColor": "#0f1117",
                        "fontWeight": "bold",
                        "color": "#e2e8f0",
                        "borderBottom": "2px solid #7c3aed",
                    },
                    style_table={"overflowX": "auto"},
                ),
                className="glass-card",
                style={"overflow": "hidden"}
            )
        ]
    )

