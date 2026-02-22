from __future__ import annotations

import importlib.util

import pytest

from veldra.gui.pages import (
    compare_page,
    config_page,
    data_page,
    results_page,
    run_page,
    runs_page,
    studio_page,
    target_page,
    train_page,
    validation_page,
)

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


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


@pytest.mark.parametrize(
    ("page_key", "layout_factory"),
    [
        ("data", data_page.layout),
        ("target", lambda: target_page.layout({})),
        ("config", config_page.layout),
        ("validation", lambda: validation_page.layout({})),
        ("train", lambda: train_page.layout({})),
        ("run", lambda: run_page.layout({})),
        ("results", results_page.layout),
        ("runs", runs_page.layout),
        ("compare", compare_page.layout),
    ],
)
def test_guided_mode_banner_present(page_key: str, layout_factory) -> None:
    layout = layout_factory()
    banner = _find_component_by_id(layout, f"guided-mode-banner-{page_key}")
    open_studio = _find_component_by_id(layout, f"guided-mode-open-studio-{page_key}")

    assert banner is not None
    assert open_studio is not None
    assert getattr(open_studio, "href", None) == "/studio"


def test_studio_page_does_not_render_guided_mode_banner() -> None:
    studio_layout = studio_page.layout()
    for key in (
        "data",
        "target",
        "config",
        "validation",
        "train",
        "run",
        "results",
        "runs",
        "compare",
    ):
        assert _find_component_by_id(studio_layout, f"guided-mode-banner-{key}") is None
