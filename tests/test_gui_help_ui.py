from __future__ import annotations

import importlib.util

import pytest

from veldra.gui.components.help_texts import HELP_TEXTS
from veldra.gui.components.help_ui import (
    context_card,
    guide_alert,
    help_icon,
    recommendation_badge,
)

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def test_help_text_entries_have_required_fields() -> None:
    assert HELP_TEXTS
    for key, value in HELP_TEXTS.items():
        assert "short" in value, key
        assert "detail" in value, key
        assert value["short"]
        assert value["detail"]


def test_help_ui_components_render() -> None:
    icon = help_icon("task_type_binary")
    assert icon is not None

    card = context_card("Title", "Body")
    assert "Title" in str(card)

    badge = recommendation_badge("Use Stratified", "info")
    assert badge is not None

    alert = guide_alert(["line1", "line2"], severity="warning")
    assert "line1" in str(alert)
