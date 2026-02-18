from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto


@pytest.mark.gui_e2e
def test_uc03_frontier_alpha_guide_visible(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/target")
    assert_ids(page, ["target-task-type", "target-frontier-alpha-guide"])
    page.get_by_text("Frontier").first.click()
    page.wait_for_selector("#target-frontier-alpha-guide")
    assert "alpha" in page.content().lower()
