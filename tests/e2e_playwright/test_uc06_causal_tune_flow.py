from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto


@pytest.mark.gui_e2e
def test_uc06_causal_tune_widgets(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/train")
    assert_ids(
        page,
        [
            "train-preset-conservative-btn",
            "train-preset-balanced-btn",
            "train-tune-objective",
            "train-objective-help",
        ],
    )
