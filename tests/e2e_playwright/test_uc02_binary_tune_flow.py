from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto


@pytest.mark.gui_e2e
def test_uc02_binary_tune_controls_visible(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/target")
    assert_ids(page, ["target-task-type", "target-task-context"])
    page.get_by_text("Binary").first.click()

    goto(page, gui_base_url, "/train")
    assert_ids(
        page,
        [
            "train-tune-enabled",
            "train-tune-preset",
            "train-tune-trials",
            "train-tune-objective",
            "train-objective-help",
        ],
    )
