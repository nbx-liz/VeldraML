from __future__ import annotations

from pathlib import Path

import pytest

from ._helpers import assert_ids, goto, set_input_value


@pytest.mark.gui_e2e
@pytest.mark.gui_smoke
def test_uc10_export_html_excel_actions(
    page,
    gui_base_url: str,
    phase26_entries: dict[str, dict],
) -> None:
    artifact_path = str(phase26_entries["UC-1"]["artifact_path"])
    artifact_root = str(Path(artifact_path).parent)

    goto(page, gui_base_url, "/results")
    assert_ids(
        page,
        [
            "artifact-root-path",
            "artifact-refresh-btn",
            "artifact-select",
            "result-export-excel-btn",
            "result-export-html-btn",
            "result-export-status",
        ],
    )

    set_input_value(page, "#artifact-root-path", artifact_root)
    page.click("#artifact-refresh-btn")
    page.wait_for_timeout(700)
    page.select_option("#artifact-select", artifact_path)

    page.click("#result-export-html-btn")
    page.wait_for_timeout(800)
    assert page.locator("#result-export-status").inner_text().strip() != ""
