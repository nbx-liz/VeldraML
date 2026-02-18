from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import html

from veldra.gui.components import charts, kpi_cards, toast


def test_charts_plotting():
    # Feature Importance
    fig = charts.plot_feature_importance({"a": 0.1, "b": 0.9})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    fig_empty = charts.plot_feature_importance({})
    assert isinstance(fig_empty, go.Figure)
    assert not fig_empty.data  # Empty

    # Metrics Bar
    fig_metrics = charts.plot_metrics_bar({"acc": 0.9, "ign": "ignore"})
    assert isinstance(fig_metrics, go.Figure)

    # Comparison
    fig_comp = charts.plot_comparison_bar({"acc": 0.8}, {"acc": 0.9})
    assert isinstance(fig_comp, go.Figure)
    assert len(fig_comp.data) == 2

    # Actual vs Pred
    fig_avp = charts.plot_actual_vs_predicted([1, 2], [1, 2])
    assert isinstance(fig_avp, go.Figure)


def test_kpi_cards():
    card = kpi_cards.kpi_card("Label", 100)
    assert isinstance(card, html.Div)
    # Check "Label" is in children?
    # Dash components structure is nested.

    card_trend = kpi_cards.kpi_card("Label", 100, trend="10%", trend_direction="up")
    assert isinstance(card_trend, html.Div)


def test_toast():
    t = toast.make_toast("message", icon="success")
    assert isinstance(t, dbc.Toast)
    assert t.children == "message"


def test_charts_learning_curves_and_empty_cases() -> None:
    fig_empty_metrics = charts.plot_metrics_bar({"label": "x"})
    assert isinstance(fig_empty_metrics, go.Figure)
    assert len(fig_empty_metrics.data) == 0

    fig_empty_comp = charts.plot_comparison_bar({"label": "x"}, {"name": "y"})
    assert isinstance(fig_empty_comp, go.Figure)
    assert len(fig_empty_comp.data) == 0

    fig_not_dict = charts.plot_learning_curves([1, 2, 3])
    assert isinstance(fig_not_dict, go.Figure)

    fig_not_list = charts.plot_learning_curves({"folds": {"x": 1}})
    assert isinstance(fig_not_list, go.Figure)

    fig_curve = charts.plot_learning_curves(
        {
            "folds": [
                {"metrics": {"rmse": [0.8, 0.6, 0.4]}},
                {"metrics": {"rmse": [0.9, 0.7, 0.5]}},
                {"metrics": {"rmse": "not-a-list"}},
                "not-a-dict",
            ]
        }
    )
    assert isinstance(fig_curve, go.Figure)
    assert len(fig_curve.data) == 3


def test_toast_variants_and_container() -> None:
    container = toast.toast_container()
    assert isinstance(container, html.Div)
    assert container.id == "toast-container"

    danger = toast.make_toast("danger", icon="danger")
    warning = toast.make_toast("warning", icon="warning")
    assert isinstance(danger, dbc.Toast)
    assert isinstance(warning, dbc.Toast)
    assert danger.icon == "danger"
    assert warning.icon == "warning"
