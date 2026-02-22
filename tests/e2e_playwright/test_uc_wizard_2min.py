"""GUI E2E legacy flow: wizard happy path smoke."""

from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto

pytestmark = [pytest.mark.gui_e2e]


def test_wizard_2min(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/train")
    assert_ids(page, ["train-wizard-open-btn"])
    page.get_by_text("Open Wizard").first.click()
    assert_ids(page, ["train-wizard-task", "train-wizard-apply-btn"])
    page.locator("#train-wizard-apply-btn").click()
    assert "train" in page.url or "run" in page.url
