from __future__ import annotations

import importlib.util

import pytest

from veldra.gui import app as app_module
from veldra.gui.pages import validation_page

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


def test_validation_layout_has_ids() -> None:
    layout = validation_page.layout({"split_config": {"type": "kfold"}})
    ids: set[str] = set()
    _collect_ids(layout, ids)
    assert "validation-split-type" in ids
    assert "validation-n-splits" in ids
    assert "validation-group-container" in ids
    assert "validation-guardrail-container" in ids


def test_save_validation_state() -> None:
    state = app_module._cb_save_validation_state(
        "timeseries",
        5,
        42,
        None,
        "dt",
        "expanding",
        20,
        1,
        0,
        {"task_type": "regression"},
    )
    split = state["split_config"]
    assert split["type"] == "timeseries"
    assert split["time_col"] == "dt"


def test_validation_split_visibility() -> None:
    group_style, ts_style = app_module._cb_update_split_options("group")
    assert group_style["display"] == "block"
    assert ts_style["display"] == "none"
