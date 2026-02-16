"""Chart generation helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go

COLORS = {
    "background": "#0f1117",
    "paper": "rgba(0,0,0,0)",
    "text": "#e2e8f0",
    "grid": "#334155",
    "accent": "#7c3aed",
    "secondary": "#06b6d4",
}


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply dark theme to plotly figure."""
    fig.update_layout(
        plot_bgcolor=COLORS["paper"],
        paper_bgcolor=COLORS["paper"],
        font={"color": COLORS["text"], "family": "Inter"},
        xaxis={"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"], "showgrid": True},
        yaxis={"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"], "showgrid": True},
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        hoverlabel={"bgcolor": "#1e293b", "font_size": 14, "font_family": "Inter"},
    )
    return fig


def plot_feature_importance(
    importance_dict: dict[str, float], title: str = "Feature Importance"
) -> go.Figure:
    """Create feature importance bar chart."""
    if not importance_dict:
        return go.Figure()

    df = (
        pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Importance"])
        .sort_values("Importance", ascending=True)
        .tail(20)
    )  # Top 20

    fig = go.Figure(
        go.Bar(
            x=df["Importance"],
            y=df["Feature"],
            orientation="h",
            marker={"color": df["Importance"], "colorscale": "Viridis", "showscale": False},
        )
    )

    fig.update_layout(title=title)
    return _apply_theme(fig)


def plot_actual_vs_predicted(
    y_true: list[float], y_pred: list[float], title: str = "Actual vs Predicted"
) -> go.Figure:
    """Create scatter plot for Actual vs Predicted."""
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))

    fig = go.Figure()

    # Reference line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"color": "#64748b", "dash": "dash"},
            name="Perfect Prediction",
            showlegend=False,
        )
    )

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker={
                "color": COLORS["accent"],
                "size": 6,
                "opacity": 0.7,
                "line": {"width": 1, "color": "white"},
            },
            name="Data Points",
        )
    )

    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted")
    return _apply_theme(fig)


def plot_metrics_bar(metrics: dict[str, float], title: str = "Metrics") -> go.Figure:
    """Create bar chart for metrics."""
    # Filter for numeric
    data = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not data:
        return go.Figure()

    fig = go.Figure(
        go.Bar(
            x=list(data.values()),
            y=list(data.keys()),
            orientation="h",
            marker={"color": COLORS["accent"]},
        )
    )
    fig.update_layout(title=title)
    return _apply_theme(fig)


def plot_comparison_bar(
    metrics1: dict[str, float],
    metrics2: dict[str, float],
    name1: str = "Current",
    name2: str = "Baseline",
) -> go.Figure:
    """Create grouped bar chart for comparison."""
    # Filter numeric
    m1 = {k: v for k, v in metrics1.items() if isinstance(v, (int, float))}
    m2 = {k: v for k, v in metrics2.items() if isinstance(v, (int, float))}

    keys = sorted(list(set(m1.keys()) | set(m2.keys())))
    if not keys:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=name1,
            y=keys,
            x=[m1.get(k, 0) for k in keys],
            orientation="h",
            marker={"color": COLORS["accent"]},
        )
    )
    fig.add_trace(
        go.Bar(
            name=name2,
            y=keys,
            x=[m2.get(k, 0) for k in keys],
            orientation="h",
            marker={"color": COLORS["secondary"]},
        )
    )

    fig.update_layout(barmode="group", title="Metric Comparison")
    return _apply_theme(fig)


def plot_learning_curves(training_history: dict[str, object]) -> go.Figure:
    """Plot fold-level learning curves from training_history payload."""
    fig = go.Figure()
    if not isinstance(training_history, dict):
        return _apply_theme(fig)

    folds = training_history.get("folds")
    if not isinstance(folds, list):
        return _apply_theme(fig)

    mean_curve: dict[int, list[float]] = {}
    for idx, fold in enumerate(folds):
        if not isinstance(fold, dict):
            continue
        metrics = fold.get("metrics")
        if not isinstance(metrics, dict):
            continue
        first_metric = next(iter(metrics.values()), None)
        if not isinstance(first_metric, list):
            continue
        xs = list(range(1, len(first_metric) + 1))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=first_metric,
                mode="lines",
                line={"width": 1},
                opacity=0.35,
                name=f"fold_{idx + 1}",
                showlegend=False,
            )
        )
        for x, y in zip(xs, first_metric):
            mean_curve.setdefault(x, []).append(float(y))

    if mean_curve:
        xs = sorted(mean_curve.keys())
        ys = [sum(mean_curve[x]) / len(mean_curve[x]) for x in xs]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"width": 3, "color": COLORS["secondary"]},
                name="mean",
            )
        )

    fig.update_layout(title="Learning Curves", xaxis_title="Iteration", yaxis_title="Metric")
    return _apply_theme(fig)
