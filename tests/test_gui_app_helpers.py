from __future__ import annotations

import importlib.util

import pytest

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")

# Obsolete tests commented out due to heavy refactoring in Phase 2
# Tests need to be rewritten for new callback structure.


def test_render_page_routes() -> None:
    from veldra.gui.app import render_page

    assert render_page("/") is not None
    assert render_page("/studio") is not None
    assert render_page("/config") is not None
    assert render_page("/run") is not None
    assert render_page("/results") is not None  # Changed artifacts to results
    assert render_page("/unknown") is not None


def _collect_text(component, out: list[str]) -> None:
    if component is None:
        return
    children = getattr(component, "children", None)
    if isinstance(children, str):
        out.append(children)
        return
    if isinstance(children, list):
        for child in children:
            _collect_text(child, out)
        return
    _collect_text(children, out)


def test_sidebar_has_three_navigation_sections() -> None:
    from veldra.gui.app import _sidebar

    sidebar = _sidebar()
    texts: list[str] = []
    _collect_text(sidebar, texts)
    joined = " ".join(texts)

    assert "Studio Mode" in joined
    assert "Guided Mode" in joined
    assert "Operations" in joined
