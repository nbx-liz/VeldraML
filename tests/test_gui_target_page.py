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
    assert "target-task-context" in ids
    assert "target-causal-context" in ids


def test_populate_target_page_infers_task(monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: lambda _p: pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
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


def test_target_task_guides_frontier() -> None:
    task_card, alpha_card = app_module._cb_target_task_guides("frontier")
    assert "Frontier" in str(task_card)
    assert "alpha" in str(alpha_card).lower()


def test_target_causal_guides() -> None:
    hint, card = app_module._cb_target_causal_guides(True, "dr_did")
    assert "DR-DiD" in hint
    assert "before/after" in str(card).lower() or "dr-did" in str(card).lower()


def test_target_guardrails_warn_for_missing_treatment(monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: lambda _p: pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
    )
    rendered = app_module._cb_target_guardrails(
        "y",
        "binary",
        [],
        True,
        "dr",
        None,
        None,
        {"data_path": "x.csv"},
    )
    assert "Treatment" in str(rendered)
