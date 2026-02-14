from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import plotly.graph_objects as go

from veldra.gui import app as app_module
from veldra.gui.types import ArtifactSummary


def test_list_artifacts_callback_and_autoselect(monkeypatch) -> None:
    items = [
        ArtifactSummary(
            path="artifacts/a",
            run_id="r1",
            task_type="regression",
            created_at_utc="2026-01-01T00:00:00+00:00",
        )
    ]
    monkeypatch.setattr(app_module, "list_artifacts", lambda _root: items)
    opts1, opts2 = app_module._cb_list_artifacts(1, "/results", "artifacts")
    assert opts1 and opts2
    assert opts1[0]["value"] == "artifacts/a"

    monkeypatch.setattr(
        app_module,
        "list_artifacts",
        lambda _root: (_ for _ in ()).throw(ValueError("x")),
    )
    assert app_module._cb_list_artifacts(1, "/results", "artifacts") == ([], [])


def test_update_result_view_empty_and_compare_fallback(monkeypatch) -> None:
    empty = app_module._cb_update_result_view(None, None)
    assert empty[0] == ""
    assert isinstance(empty[1], go.Figure)
    assert "Select an artifact" in empty[3]

    class _Artifact:
        def __init__(self) -> None:
            self.metrics = {"r2": 0.8, "mae": 0.1, "extra": 1.2}
            self.feature_schema = {"feature_importance": {"x1": 1.0}}
            self.manifest = SimpleNamespace(
                run_id="rid", task_type="regression", created_at_utc=None
            )
            self.run_config = {"task": {"type": "regression"}}

    def _load(path: str):
        if "compare" in path:
            raise RuntimeError("compare load failed")
        return _Artifact()

    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=_load))
    monkeypatch.setattr(
        app_module,
        "plot_metrics_bar",
        lambda m, title=None: {"title": title, "m": m},
    )
    monkeypatch.setattr(app_module, "plot_comparison_bar", lambda *a, **k: {"kind": "comp"})
    monkeypatch.setattr(app_module, "plot_feature_importance", lambda fi: {"fi": fi})
    monkeypatch.setattr(app_module, "kpi_card", lambda k, v: f"{k}:{v}")
    monkeypatch.setattr(app_module, "_json_dumps", lambda payload: "cfg")
    monkeypatch.setattr(app_module, "_format_jst_timestamp", lambda value: "ts")

    kpi, fig_main, fig_sec, details = app_module._cb_update_result_view("main", "compare")
    assert "extra" in str(kpi)
    assert fig_main["title"] == "Performance Metrics"
    assert fig_sec["fi"]["x1"] == 1.0
    assert "Run ID" in str(details)


def test_update_result_view_feature_importance_booster_fallback(monkeypatch) -> None:
    class _Booster:
        def feature_importance(self, importance_type="gain"):  # noqa: ARG002
            return [2.0, 0.0]

        def feature_name(self):
            return ["x1", "x2"]

    class _Artifact:
        def __init__(self) -> None:
            self.metrics = {"mean": {"rmse": 1.0}}
            self.feature_schema = {"feature_names": ["x1", "x2"]}
            self.manifest = SimpleNamespace(
                run_id="rid", task_type="regression", created_at_utc=None
            )
            self.run_config = {"task": {"type": "regression"}}

        def _get_booster(self):
            return _Booster()

    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=lambda _path: _Artifact()))
    monkeypatch.setattr(
        app_module,
        "plot_metrics_bar",
        lambda m, title=None: {"m": m, "title": title},
    )
    monkeypatch.setattr(app_module, "plot_feature_importance", lambda fi: {"fi": fi})
    monkeypatch.setattr(app_module, "kpi_card", lambda k, v: f"{k}:{v}")
    monkeypatch.setattr(app_module, "_json_dumps", lambda payload: "cfg")
    monkeypatch.setattr(app_module, "_format_jst_timestamp", lambda value: "ts")

    _, _, fig_sec, _ = app_module._cb_update_result_view("main", None)
    assert fig_sec["fi"] == {"x1": 2.0}


def test_update_result_view_load_error(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "Artifact",
        SimpleNamespace(load=lambda _path: (_ for _ in ()).throw(RuntimeError("load failed"))),
    )
    res = app_module._cb_update_result_view("bad", None)
    assert "Error loading artifact" in str(res[0])


def test_result_evaluate_action_error_branch(monkeypatch) -> None:
    @dataclass
    class _Eval:
        m: int

    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=lambda _p: object()))
    monkeypatch.setattr(app_module, "_get_load_tabular_data", lambda: (lambda _p: object()))
    monkeypatch.setattr(app_module, "_get_evaluate", lambda: (lambda _a, _d: _Eval(m=1)))
    ok = app_module._cb_evaluate_artifact_action(1, "a", "d")
    assert '"m": 1' in ok

    monkeypatch.setattr(
        app_module,
        "_get_evaluate",
        lambda: (lambda _a, _d: (_ for _ in ()).throw(RuntimeError("eval failed"))),
    )
    err = app_module._cb_evaluate_artifact_action(1, "a", "d")
    assert "Evaluation failed" in err
