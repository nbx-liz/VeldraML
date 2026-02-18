from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto


@pytest.mark.gui_e2e
@pytest.mark.gui_smoke
def test_uc01_regression_navigation_flow(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/data")
    page.wait_for_selector("#data-upload-drag", state="visible")
    assert_ids(page, ["data-upload-drag"])

    goto(page, gui_base_url, "/target")
    assert_ids(page, ["target-col-select", "target-task-type", "target-guardrail-container"])

    goto(page, gui_base_url, "/validation")
    assert_ids(page, ["validation-split-type", "validation-guardrail-container"])

    goto(page, gui_base_url, "/train")
    assert_ids(page, ["train-learning-rate", "train-config-yaml-preview"])

    goto(page, gui_base_url, "/run")
    assert_ids(
        page,
        ["run-action-display", "run-guardrail-container", "run-execute-btn", "run-priority"],
    )

    goto(page, gui_base_url, "/results")
    assert_ids(page, ["artifact-select", "artifact-evaluate-btn", "result-export-excel-btn"])
