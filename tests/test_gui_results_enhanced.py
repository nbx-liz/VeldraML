from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from veldra.gui import app as app_module
from veldra.gui.pages import results_page

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


def test_results_layout_has_new_tabs() -> None:
    layout = results_page.layout()
    ids: set[str] = set()
    _collect_ids(layout, ids)
    assert "result-learning-curve" in ids
    assert "result-config-view" in ids
    assert "result-export-excel-btn" in ids


def test_update_result_extras(monkeypatch, tmp_path) -> None:
    artifact_path = tmp_path / "art"
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "training_history.json").write_text('{"folds": []}', encoding="utf-8")

    fake_art = SimpleNamespace(
        metadata={},
        config={"task": {"type": "regression"}},
        run_config={"task": {"type": "regression"}},
        feature_schema={"feature_names": ["a", "b"], "n_rows": 10},
        task_type="regression",
    )
    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=lambda _p: fake_art))

    fig, cfg, summary = app_module._cb_update_result_extras(str(artifact_path))
    assert hasattr(fig, "to_dict")
    assert "task" in cfg.lower()
    assert "Features" in str(summary)
