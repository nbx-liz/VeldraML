from __future__ import annotations
from veldra.gui.pages import data_page
from dash import html

def test_render_data_stats():
    stats = {
        "n_rows": 10,
        "n_cols": 2,
        "numeric_cols": ["a"],
        "categorical_cols": ["b"],
        "missing_count": 0,
        "columns": ["a", "b"]
    }
    div = data_page.render_data_stats(stats)
    assert isinstance(div, html.Div)

def test_render_data_preview():
    # Empty
    assert isinstance(data_page.render_data_preview([]), html.Div)
    
    # Valid
    preview = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    div = data_page.render_data_preview(preview)
    assert isinstance(div, html.Div)
