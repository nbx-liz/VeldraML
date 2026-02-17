from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto


@pytest.mark.gui_e2e
def test_uc09_export_controls_and_onnx_option(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/run")
    assert_ids(
        page,
        [
            "run-action-override-mode",
            "run-action-manual",
            "run-export-format",
            "run-artifact-path",
        ],
    )

    page.click("#run-action-override-mode input[value='manual']")
    page.select_option("#run-action-manual", "export")
    page.select_option("#run-export-format", "onnx")

    assert "EXPORT" in page.locator("#run-action-display").inner_text().upper()
