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

    assert render_page("/config") is not None
    assert render_page("/run") is not None
    assert render_page("/results") is not None # Changed artifacts to results
    assert render_page("/unknown") is not None
