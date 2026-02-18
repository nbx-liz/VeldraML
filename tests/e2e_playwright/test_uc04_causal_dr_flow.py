from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto


@pytest.mark.gui_e2e
def test_uc04_causal_dr_guidance(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/target")
    assert_ids(page, ["target-causal-enabled", "target-causal-method", "target-causal-context"])
    page.get_by_text("Causal Settings").first.click()
    page.check("#target-causal-enabled")
    page.select_option("#target-causal-method", "dr")
    assert "dr" in page.content().lower()
