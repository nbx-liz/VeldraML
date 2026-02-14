from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from veldra.gui.pages import config_page, results_page, run_page

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def test_page_layouts_have_expected_component_ids() -> None:
    config = config_page.layout()
    run = run_page.layout()
    results = results_page.layout()

    def _collect_ids(component, out: set[str]) -> None:
        if component is None:
            return
        component_id = getattr(component, "id", None)
        if isinstance(component_id, str):
            out.add(component_id)
        children = getattr(component, "children", None)
        if isinstance(children, list):
            for child in children:
                _collect_ids(child, out)
        else:
            _collect_ids(children, out)

    ids: set[str] = set()
    _collect_ids(config, ids)
    _collect_ids(run, ids)
    _collect_ids(results, ids)
    assert "config-yaml" in ids
    assert "config-migrate-preview-btn" in ids
    assert "run-action" in ids
    assert "run-job-select" in ids
    assert "artifact-root-path" in ids


def test_gui_init_create_app_proxy(monkeypatch) -> None:
    import veldra.gui as gui

    fake_module = types.ModuleType("veldra.gui.app")
    fake_module.create_app = lambda: "ok-app"
    monkeypatch.setitem(sys.modules, "veldra.gui.app", fake_module)
    assert gui.create_app() == "ok-app"


def test_config_builder_ux_layout_contracts() -> None:
    builder = config_page._render_builder_tab()
    data_settings_card = builder.children[5]
    split_strategy_card = builder.children[7]

    def _collect_ids(component, out: set[str]) -> None:
        if component is None:
            return
        component_id = getattr(component, "id", None)
        if isinstance(component_id, str):
            out.add(component_id)
        children = getattr(component, "children", None)
        if isinstance(children, list):
            for child in children:
                _collect_ids(child, out)
        else:
            _collect_ids(children, out)

    data_ids: set[str] = set()
    split_ids: set[str] = set()
    _collect_ids(data_settings_card, data_ids)
    _collect_ids(split_strategy_card, split_ids)

    assert "cfg-container-id-cols" in data_ids
    assert "cfg-container-id-cols" not in split_ids

    def _find_component_by_id(component, component_id: str):
        if component is None:
            return None
        if getattr(component, "id", None) == component_id:
            return component
        children = getattr(component, "children", None)
        if isinstance(children, list):
            for child in children:
                found = _find_component_by_id(child, component_id)
                if found is not None:
                    return found
            return None
        return _find_component_by_id(children, component_id)

    run_now_btn = _find_component_by_id(builder, "config-run-now-btn")
    to_run_btn = _find_component_by_id(builder, "config-to-run-btn")
    export_preset = _find_component_by_id(builder, "cfg-export-dir-preset")
    ts_warning = _find_component_by_id(builder, "cfg-timeseries-time-warning")
    ts_col = _find_component_by_id(builder, "cfg-split-time-col")

    assert run_now_btn is not None
    assert run_now_btn.color == "primary"
    assert to_run_btn is not None
    assert to_run_btn.color == "primary"
    assert export_preset is not None
    assert export_preset.value == "artifacts"
    assert ts_warning is not None
    assert ts_col is not None
    assert ts_col.placeholder == "Select required time column..."
    assert "Categorical Columns (Optional override)" not in str(data_settings_card)
    assert "ID Columns (Optional - for Group K-Fold)" not in str(split_strategy_card)
