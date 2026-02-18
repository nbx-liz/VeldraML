from __future__ import annotations

from veldra.gui.pages import run_page


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


def test_run_page_layout_defaults() -> None:
    layout = run_page.layout()
    ids: set[str] = set()
    _collect_ids(layout, ids)
    assert "run-data-path" in ids
    assert "run-config-path" in ids
    assert "run-execute-btn" in ids

    data_input = _find_component_by_id(layout, "run-data-path")
    config_input = _find_component_by_id(layout, "run-config-path")
    execute_btn = _find_component_by_id(layout, "run-execute-btn")

    assert data_input is not None
    assert config_input is not None
    assert execute_btn is not None
    assert data_input.value == ""
    assert config_input.value == "configs/gui_run.yaml"
    assert execute_btn.disabled is True


def test_run_page_layout_respects_state_and_config_yaml_none() -> None:
    layout = run_page.layout({"data_path": "data/train.csv", "config_yaml": None})

    data_input = _find_component_by_id(layout, "run-data-path")
    execute_btn = _find_component_by_id(layout, "run-execute-btn")
    status_alert = _find_component_by_id(layout, "run-launch-status")

    assert data_input is not None
    assert execute_btn is not None
    assert status_alert is not None
    assert data_input.value == "data/train.csv"
    assert execute_btn.disabled is False
    assert "Ready" in str(status_alert.children)
