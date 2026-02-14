from __future__ import annotations

import pytest
from veldra.gui.components import charts, toast, kpi_cards
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go

def test_charts_plotting():
    # Feature Importance
    fig = charts.plot_feature_importance({"a": 0.1, "b": 0.9})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    
    fig_empty = charts.plot_feature_importance({})
    assert isinstance(fig_empty, go.Figure)
    assert not fig_empty.data # Empty

    # Metrics Bar
    fig_metrics = charts.plot_metrics_bar({"acc": 0.9, "ign": "ignore"})
    assert isinstance(fig_metrics, go.Figure)
    
    # Comparison
    fig_comp = charts.plot_comparison_bar({"acc": 0.8}, {"acc": 0.9})
    assert isinstance(fig_comp, go.Figure)
    assert len(fig_comp.data) == 2

    # Actual vs Pred
    fig_avp = charts.plot_actual_vs_predicted([1,2], [1,2])
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
