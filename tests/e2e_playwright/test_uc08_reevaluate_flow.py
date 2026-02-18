from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto, set_input_value


@pytest.mark.gui_e2e
@pytest.mark.gui_smoke
def test_uc08_precheck_before_reevaluate(
    page,
    gui_base_url: str,
    artifact_path_uc1: str,
    artifact_root_uc1: str,
    sample_data_paths: dict[str, str],
) -> None:
    goto(page, gui_base_url, "/results")
    assert_ids(
        page,
        [
            "artifact-root-path",
            "artifact-refresh-btn",
            "artifact-select",
            "artifact-eval-data-path",
            "artifact-eval-precheck",
        ],
    )

    set_input_value(page, "#artifact-root-path", artifact_root_uc1)
    page.click("#artifact-refresh-btn")
    page.wait_for_timeout(700)
    page.select_option("#artifact-select", artifact_path_uc1)

    bad_path = sample_data_paths.get("uc8_bad", "")
    set_input_value(page, "#artifact-eval-data-path", bad_path)
    page.wait_for_timeout(500)

    precheck_text = page.locator("#artifact-eval-precheck").inner_text().lower()
    assert precheck_text != ""
    assert "missing" in precheck_text or "precheck" in precheck_text
