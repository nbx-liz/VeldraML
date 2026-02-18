from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto, set_input_value


@pytest.mark.gui_e2e
def test_uc07_evaluate_existing_artifact(
    page,
    gui_base_url: str,
    artifact_path_uc1: str,
    artifact_root_uc1: str,
    sample_data_paths: dict[str, str],
) -> None:
    eval_data = sample_data_paths.get("uc1_test", "")

    goto(page, gui_base_url, "/results")
    assert_ids(
        page,
        [
            "artifact-root-path",
            "artifact-refresh-btn",
            "artifact-select",
            "artifact-eval-data-path",
            "artifact-evaluate-btn",
        ],
    )

    set_input_value(page, "#artifact-root-path", artifact_root_uc1)
    page.click("#artifact-refresh-btn")
    page.wait_for_timeout(700)

    page.select_option("#artifact-select", artifact_path_uc1)
    set_input_value(page, "#artifact-eval-data-path", eval_data)
    page.click("#artifact-evaluate-btn")

    page.wait_for_selector("#artifact-eval-result")
    page.wait_for_function(
        "() => (document.querySelector('#artifact-eval-result')?.innerText || '').trim().length > 0"
    )
    result_text = page.locator("#artifact-eval-result").inner_text()
    assert result_text.strip() != ""
