from __future__ import annotations

import types
from dataclasses import dataclass

import plotly.graph_objs as go

from veldra.gui.types import PaginatedResult


# Reuse the get_callback helper (or duplicate it for independence)
def get_callback(app, output_substr: str):
    for key, value in app.callback_map.items():
        if output_substr in key:
            return value["callback"].__wrapped__
    raise KeyError(f"No callback found for output '{output_substr}'")


@dataclass
class MockArtifact:
    path: str
    run_id: str
    task_type: str = "regression"
    created_at_utc: str = "2023-01-01"
    metrics: dict | None = None
    metadata: dict | None = None
    config: dict | None = None

    @classmethod
    def load(cls, path):
        if "error" in path:
            raise ValueError("Load failed")

        metrics = {"accuracy": 0.9} if "metrics" in path else {}
        if "feature" in path:
            metadata = {"feature_importance": {"f1": 0.5}}
        else:
            metadata = {}

        return cls(path, "run_123", metrics=metrics, metadata=metadata, config={"foo": "bar"})


def test_results_callbacks(monkeypatch):
    import veldra.gui.app as app_module

    # Mock Artifact class
    monkeypatch.setattr(app_module, "Artifact", MockArtifact)

    app = app_module.create_app()

    # 1. List Artifacts
    # Output: artifact-select.options
    list_cb = get_callback(app, "artifact-select.options")

    monkeypatch.setattr(
        app_module,
        "list_artifacts_page",
        lambda **_kwargs: PaginatedResult(
            items=[MockArtifact("p1", "r1")], total_count=1, limit=50, offset=0
        ),
    )

    opts, opts_comp, page, total, info = list_cb(1, "/results", 0, 0, 50, "", "root", 0)
    assert len(opts) == 1
    assert opts[0]["value"] == "p1"
    assert page == 0
    assert total == 1
    assert "1" in info

    monkeypatch.setattr(
        app_module,
        "list_artifacts_page",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("FAIL")),
    )
    opts_err, _, _, _, _ = list_cb(1, "/results", 0, 0, 50, "", "root", 0)
    assert opts_err == []

    # 2. Update Result View
    # Output: result-chart-main.figure
    view_cb = get_callback(app, "result-chart-main.figure")

    # Simple view
    res = view_cb("path_metrics", None)
    # Returns: (kpi_container, fig_main, fig_sec, details_elem)
    assert len(res) == 4
    # KPIs
    assert len(res[0].children) > 0  # Should have accuracy
    # Main Chart
    assert res[1] is not None  # Figure
    # Secondary Chart (now returns empty Figure, not dict)
    assert isinstance(res[2], (dict, go.Figure)) and (res[2] != {} or hasattr(res[2], "to_dict"))

    # Feature Importance view
    res_feat = view_cb("path_metrics_feature", None)
    assert res_feat[2] is not None  # Should be figure

    # Feature Importance fallback from booster when metadata/schema does not include it
    original_load_with_feature = MockArtifact.load

    def load_with_booster_fi(path):
        if "boosterfi" in path:
            obj = types.SimpleNamespace(
                path=path,
                run_id="run_booster",
                task_type="regression",
                created_at_utc="2023-01-01",
                metrics={"rmse": 0.1},
                metadata={},
                feature_schema={"feature_names": ["f1", "f2"]},
                config={"foo": "bar"},
            )

            class BoosterMock:
                def feature_importance(self, importance_type="gain"):
                    return [2.5, 1.0]

                def feature_name(self):
                    return ["f1", "f2"]

            obj._get_booster = lambda: BoosterMock()
            return obj
        return original_load_with_feature(path)

    monkeypatch.setattr(MockArtifact, "load", load_with_booster_fi)
    res_booster_fi = view_cb("path_boosterfi", None)
    assert hasattr(res_booster_fi[2], "to_dict")
    assert len(res_booster_fi[2].data) > 0

    # Nested mean metrics view (artifact metrics contract)
    original_load = MockArtifact.load

    def load_with_mean(path):
        if "mean" in path:
            return MockArtifact(
                path,
                "run_mean",
                metrics={"folds": [], "mean": {"rmse": 0.1, "mae": 0.05, "r2": 0.9}},
                metadata={},
                config={},
            )
        return original_load(path)

    monkeypatch.setattr(MockArtifact, "load", load_with_mean)
    res_mean = view_cb("path_mean_metrics", None)
    assert len(res_mean[1].data) > 0

    # Comparison view
    res_comp = view_cb("path_metrics", "path_metrics")  # Same path -> no comp
    assert "Comparison" not in str(res_comp[1])  # Maybe check layout title?

    # Different path
    # MockArtifact.load logic relies on path string to vary content?
    # Our MockArtifact.load returns same structure mostly.
    # To test comparison, we need different metrics.
    # Modify MockArtifact.load to return different metrics based on path?
    # "path_metrics" -> acc 0.9
    # "path_metrics_2" -> acc 0.8

    def side_effect_load(path):
        if "error" in path:
            raise ValueError()
        metrics = {"accuracy": 0.9}
        if "2" in path:
            metrics = {"accuracy": 0.8}
        return MockArtifact(path, "r1", metrics=metrics, config={})

    monkeypatch.setattr(MockArtifact, "load", side_effect_load)

    res_comp_2 = view_cb("path_metrics", "path_metrics_2")

    # Debugging
    if isinstance(res_comp_2[1], dict) and not res_comp_2[1]:
        # Empty dict means error
        raise AssertionError(f"Callback returned error: {res_comp_2[0]}")

    # plot_comparison_bar returns Figure object
    assert len(res_comp_2[1].data) == 2

    # No artifact
    res_none = view_cb(None, None)
    assert res_none[0] == ""

    # Error
    res_err = view_cb("error_path", None)
    assert "Error" in res_err[0].children

    # 3. Evaluate Artifact
    # Output: artifact-eval-result.children
    eval_cb = get_callback(app, "artifact-eval-result.children")

    # _evaluate_artifact_action calls load_tabular_data AND evaluate DIRECTLY
    from veldra.api.types import EvalResult

    monkeypatch.setattr(app_module, "load_tabular_data", lambda p: "data_frame")
    monkeypatch.setattr(
        app_module,
        "evaluate",
        lambda a, d: EvalResult(task_type="regression", metrics={"score": 0.95}, metadata={}),
    )

    res_eval = eval_cb(1, "path", "data_path")
    assert '"score": 0.95' in res_eval

    # Missing args
    assert "required" in eval_cb(1, None, None)

    # Error
    monkeypatch.setattr(
        app_module, "evaluate", lambda a, d: (_ for _ in ()).throw(ValueError("boom"))
    )
    assert "boom" in eval_cb(1, "p", "d")
