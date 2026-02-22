"""GUI E2E legacy flow: template apply then run handoff."""

from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto

pytestmark = [pytest.mark.gui_e2e]


def test_template_to_run(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/train")
    assert_ids(
        page,
        [
            "train-template-select",
            "train-template-apply-btn",
            "train-slot-save-btn",
            "train-diff-view",
        ],
    )
    page.get_by_text("Apply Template").first.click()
    goto(page, gui_base_url, "/run")
    assert_ids(page, ["run-execute-btn", "run-config-yaml"])
