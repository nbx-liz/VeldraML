from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from veldra.gui import app as app_module
from veldra.gui.pages import compare_page

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


def test_compare_layout_has_ids() -> None:
    layout = compare_page.layout()
    ids: set[str] = set()
    _collect_ids(layout, ids)
    assert "compare-artifact-a" in ids
    assert "compare-artifact-b" in ids
    assert "compare-metrics-table" in ids


def test_populate_compare_options(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "list_artifacts",
        lambda _root: [SimpleNamespace(task_type="binary", run_id="r1", path="artifacts/r1")],
    )
    opts_a, opts_b, va, vb = app_module._cb_populate_compare_options(
        "/compare", {"compare_selection": ["artifacts/r1", "artifacts/r2"]}
    )
    assert len(opts_a) == 1
    assert va == "artifacts/r1"
    assert vb == "artifacts/r2"


def test_compare_runs_callback(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "compare_artifacts",
        lambda *_a: {
            "checks": [{"level": "ok", "message": "ok"}],
            "metrics_a": {"auc": 0.8},
            "metrics_b": {"auc": 0.7},
            "metric_rows": [{"metric": "auc", "run_a": 0.8, "run_b": 0.7, "delta": 0.1}],
            "config_yaml_a": "a: 1\n",
            "config_yaml_b": "a: 2\n",
        },
    )
    checks, rows, fig, diff = app_module._cb_compare_runs("a", "b")
    assert rows[0]["metric"] == "auc"
    assert hasattr(fig, "to_dict")
    assert "Run A" in str(diff)
