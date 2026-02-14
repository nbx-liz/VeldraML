"""Tests for new GUI pages (Data, Results)."""

from __future__ import annotations

import importlib.util

import pytest

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def _collect_ids(component, out: set[str]) -> None:
    if component is None:
        return
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        out.add(component_id)

    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            _collect_ids(child, out)
    else:
        _collect_ids(children, out)


def test_data_page_layout() -> None:
    from veldra.gui.pages import data_page

    layout = data_page.layout()
    ids: set[str] = set()
    _collect_ids(layout, ids)

    assert "data-file-path" in ids
    assert "data-loading" in ids


def test_results_page_layout() -> None:
    from veldra.gui.pages import results_page

    layout = results_page.layout()
    ids: set[str] = set()
    _collect_ids(layout, ids)

    assert "artifact-root-path" in ids
    assert "artifact-select" in ids
    assert "artifact-refresh-btn" in ids
    assert "result-chart-main" in ids
    assert "result-chart-secondary" in ids
