"""GUI E2E legacy flow: runs/results pagination smoke."""

from __future__ import annotations

import pytest

from ._helpers import assert_ids, goto


@pytest.mark.gui_e2e
def test_runs_results_pagination_controls(page, gui_base_url: str) -> None:
    goto(page, gui_base_url, "/runs")
    assert_ids(
        page,
        [
            "runs-page-prev-btn",
            "runs-page-next-btn",
            "runs-page-size",
            "runs-page-info",
        ],
    )

    goto(page, gui_base_url, "/results")
    assert_ids(
        page,
        [
            "artifact-page-prev-btn",
            "artifact-page-next-btn",
            "artifact-page-size-select",
            "artifact-page-info",
        ],
    )
