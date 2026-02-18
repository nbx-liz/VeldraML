from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pandas as pd
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
    assert "result-report-download" in ids
    assert "result-export-job-store" in ids
    assert "result-export-poll-interval" in ids
    assert "artifact-eval-precheck" in ids
    assert "result-export-help" in ids


def test_update_result_extras(monkeypatch, tmp_path) -> None:
    artifact_path = tmp_path / "art"
    artifact_path.mkdir(parents=True, exist_ok=True)

    fake_art = SimpleNamespace(
        training_history={"folds": []},
        config={"task": {"type": "regression"}},
        run_config={"task": {"type": "regression"}},
        feature_schema={"feature_names": ["a", "b"], "n_rows": 10},
        task_type="regression",
    )
    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=lambda _p: fake_art))

    fig, fold, causal, options, drill, cfg, summary = app_module._cb_update_result_extras(
        str(artifact_path)
    )
    assert hasattr(fig, "to_dict")
    assert hasattr(fold, "to_dict")
    assert hasattr(causal, "to_dict")
    assert isinstance(options, list)
    assert hasattr(drill, "to_dict")
    assert "task" in cfg.lower()
    assert "Features" in str(summary)


def test_result_eval_precheck_and_shortcut_highlight(monkeypatch) -> None:
    fake_art = SimpleNamespace(
        feature_schema={"feature_names": ["a", "b"]},
    )
    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=lambda _p: fake_art))
    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: lambda _p: pd.DataFrame({"a": [1], "b": [2]}),
    )
    precheck = app_module._cb_result_eval_precheck("artifacts/x", "data.csv")
    assert "passed" in str(precheck).lower()

    classes = app_module._cb_result_shortcut_highlight(
        {"results_shortcut_focus": "evaluate"},
        "/results",
    )
    assert "border-warning" in classes[0]
