from __future__ import annotations

from dash import html

from veldra.gui.pages import data_page


def _find_component_by_id(component, component_id: str):
    if component is None:
        return None
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            found = _find_component_by_id(child, component_id)
            if found is not None:
                return found
        return None
    return _find_component_by_id(children, component_id)


def _find_first_th(component):
    if component is None:
        return None
    if component.__class__.__name__ == "Th":
        return component
    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            found = _find_first_th(child)
            if found is not None:
                return found
        return None
    return _find_first_th(children)


def test_render_data_stats():
    stats = {
        "n_rows": 10,
        "n_cols": 2,
        "numeric_cols": ["a"],
        "categorical_cols": ["b"],
        "datetime_cols": [],
        "missing_count": 0,
        "columns": ["a", "b"],
        "column_profiles": [
            {
                "name": "a",
                "kind": "numeric",
                "dtype": "float64",
                "missing_rate": 0.0,
                "unique_count": 10,
            },
            {
                "name": "b",
                "kind": "categorical",
                "dtype": "object",
                "missing_rate": 0.0,
                "unique_count": 2,
            },
        ],
        "warnings": [],
    }
    div = data_page.render_data_stats(stats)
    assert isinstance(div, html.Div)
    next_btn = _find_component_by_id(div, "data-to-target-btn")
    assert next_btn is not None
    assert next_btn.color == "primary"


def test_render_data_preview():
    # Empty
    assert isinstance(data_page.render_data_preview([]), html.Div)

    # Valid
    preview = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    div = data_page.render_data_preview(preview)
    assert isinstance(div, html.Div)
    header_cell = _find_first_th(div)
    assert header_cell is not None
    assert header_cell.style["position"] == "sticky"
    assert header_cell.style["top"] == "0"
    assert header_cell.style["zIndex"] == "10"
    assert header_cell.style["backgroundColor"] == "var(--bg-secondary)"
