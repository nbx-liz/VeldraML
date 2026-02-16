from __future__ import annotations

import importlib.util

import pytest

from veldra.gui import app as app_module
from veldra.gui.pages import train_page

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def _collect_ids(component, out: set[str]) -> None:
    if component is None:
        return
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        out.add(cid)
    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            _collect_ids(child, out)
    else:
        _collect_ids(children, out)


def test_train_layout_has_builder_and_yaml() -> None:
    layout = train_page.layout({"config_yaml": "config_version: 1\n"})
    ids: set[str] = set()
    _collect_ids(layout, ids)
    assert "train-learning-rate" in ids
    assert "train-config-yaml-preview" in ids
    assert "train-config-validate-btn" in ids


def test_save_train_state_and_preview() -> None:
    state = app_module._cb_save_train_state(
        0.05,
        200,
        31,
        -1,
        20,
        50,
        True,
        "",
        False,
        "standard",
        30,
        None,
        "artifacts",
        {"task_type": "regression", "data_path": "data.csv", "target_col": "y"},
    )
    yaml_text, summary = app_module._cb_update_train_yaml_preview(state)
    assert "num_boost_round" in yaml_text
    assert "Task:" in str(summary)
