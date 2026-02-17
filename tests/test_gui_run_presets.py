from __future__ import annotations

import importlib.util

import pytest

from veldra.gui import app as app_module

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def test_detect_run_action_manual_override() -> None:
    action, text, _klass = app_module._cb_detect_run_action(
        "task:\n  type: regression\n",
        "manual",
        "evaluate",
    )
    assert action == "evaluate"
    assert "EVALUATE" in text


def test_run_action_manual_visibility_and_description() -> None:
    assert app_module._cb_run_action_manual_visibility("manual")["display"] == "block"
    desc = app_module._cb_run_action_description("fit", "manual")
    assert "MANUAL" in desc
    assert "Train model" in desc


def test_run_guardrails_and_launch_state() -> None:
    rendered, has_error = app_module._cb_run_guardrails(
        "simulate",
        "",
        "",
        "",
        "",
        "",
    )
    assert has_error is True
    assert "Scenarios Path" in str(rendered)

    launch = app_module._cb_update_run_launch_state(
        "fit",
        "data.csv",
        "config_version: 1\n",
        "",
        "",
        "",
        True,
    )
    assert launch[0] is True
    assert "Guardrail error" in launch[1]
