from __future__ import annotations

import importlib.util

import pytest

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def _find_callbacks(app, output_substr: str):
    matches = []
    for key, value in app.callback_map.items():
        if output_substr in key:
            matches.append(value)
    return matches


def test_callback_wiring_exists_for_core_outputs() -> None:
    import veldra.gui.app as app_module

    app = app_module.create_app()

    required_outputs = [
        "page-content.children",
        "stepper-content.children",
        "run-jobs-interval.interval",
        "run-jobs-table-container.children",
        "run-job-detail.children",
        "artifact-select.options",
        "result-chart-main.figure",
        "artifact-eval-result.children",
        "config-yaml.value",
        "config-migrate-normalized-yaml.value",
    ]
    for out in required_outputs:
        assert _find_callbacks(app, out), f"Missing callback wiring for output: {out}"


def test_duplicate_run_result_log_callbacks_are_distinguishable() -> None:
    import veldra.gui.app as app_module

    app = app_module.create_app()
    matches = _find_callbacks(app, "run-result-log.children")
    assert len(matches) >= 2

    has_enqueue = False
    has_cancel = False
    for cb in matches:
        inputs = cb.get("inputs", [])
        ids = {entry.get("id") for entry in inputs}
        if "run-execute-btn" in ids:
            has_enqueue = True
        if "run-cancel-job-btn" in ids:
            has_cancel = True

    assert has_enqueue is True
    assert has_cancel is True
