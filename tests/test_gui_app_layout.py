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


def test_gui_layout_contains_core_regions() -> None:
    from veldra.gui.app import create_app

    app = create_app()
    ids: set[str] = set()
    _collect_ids(app.layout, ids)
    assert "url" in ids
    assert "page-content" in ids
    assert "gui-global-message" in ids
