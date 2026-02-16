from __future__ import annotations

import importlib.util

import pytest

from veldra.gui import app as app_module
from veldra.gui.pages import target_page

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


def test_target_layout_has_core_ids() -> None:
    layout = target_page.layout({"task_type": "binary"})
    ids: set[str] = set()
    _collect_ids(layout, ids)
    assert "target-col-select" in ids
    assert "target-task-type" in ids
    assert "target-exclude-cols" in ids
    assert "target-guardrail-container" in ids


def test_populate_target_page_infers_task(monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: (lambda _p: pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})),
    )
    out = app_module._cb_populate_target_page("/target", {"data_path": "x.csv", "target_col": "y"})
    assert out[2] == "y"
    assert out[7] == "binary"


def test_save_target_state_updates_yaml() -> None:
    state = app_module._cb_save_target_state(
        "target",
        "regression",
        ["drop_a"],
        False,
        None,
        None,
        None,
        {"data_path": "data.csv"},
    )
    assert state["target_col"] == "target"
    assert "config_yaml" in state
