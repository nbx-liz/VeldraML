from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from veldra.diagnostics.plots import (
    plot_error_histogram,
    plot_feature_importance,
    plot_frontier_scatter,
    plot_lift_chart,
    plot_nll_histogram,
    plot_pinball_histogram,
    plot_roc_comparison,
    plot_shap_summary,
    plot_timeseries_prediction,
    plot_timeseries_residual,
    plot_true_class_prob_histogram,
)


def _assert_file(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_functions_create_png_files(tmp_path) -> None:
    plot_error_histogram(np.random.randn(20), np.random.randn(20), {}, {}, tmp_path / "err.png")
    plot_roc_comparison(
        [0, 1, 0, 1],
        [0.1, 0.8, 0.3, 0.7],
        [0, 1, 0, 1],
        [0.2, 0.7, 0.4, 0.6],
        tmp_path / "roc.png",
    )
    plot_lift_chart([0, 1, 0, 1], [0.2, 0.9, 0.3, 0.8], tmp_path / "lift.png")
    plot_nll_histogram(np.random.rand(20), np.random.rand(20), tmp_path / "nll.png")
    plot_true_class_prob_histogram(np.random.rand(20), np.random.rand(20), tmp_path / "prob.png")
    plot_timeseries_prediction(
        np.arange(20),
        np.random.rand(20),
        np.random.rand(20),
        10,
        tmp_path / "ts_pred.png",
    )
    plot_timeseries_residual(np.arange(20), np.random.randn(20), 10, tmp_path / "ts_res.png")
    plot_pinball_histogram(np.random.rand(20), np.random.rand(20), tmp_path / "pinball.png")
    plot_frontier_scatter(np.random.rand(20), np.random.rand(20), tmp_path / "frontier.png")
    plot_feature_importance(
        pd.DataFrame({"feature": ["a", "b"], "importance": [1.0, 2.0]}),
        "gain",
        tmp_path / "importance.png",
    )
    plot_shap_summary(
        pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)}),
        pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)}),
        tmp_path / "shap.png",
    )

    for name in [
        "err.png",
        "roc.png",
        "lift.png",
        "nll.png",
        "prob.png",
        "ts_pred.png",
        "ts_res.png",
        "pinball.png",
        "frontier.png",
        "importance.png",
        "shap.png",
    ]:
        _assert_file(tmp_path / name)
