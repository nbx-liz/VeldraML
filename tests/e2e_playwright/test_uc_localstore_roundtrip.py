"""GUI E2E legacy flow: local-store save/load roundtrip."""

from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto

pytestmark = [pytest.mark.gui_e2e]


def test_localstore_roundtrip(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/train")
    assert_ids(page, ["train-slot-save-btn", "train-slot-select"])
    page.get_by_text("Save Slot").first.click()
    page.reload()
    assert_ids(page, ["train-slot-select", "train-slot-load-btn"])
