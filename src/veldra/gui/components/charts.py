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


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
            }
        ],
    )
    return _apply_theme(fig)


def plot_feature_importance(
    importance_dict: dict[str, float], title: str = "Feature Importance"
) -> go.Figure:
    """Create feature importance bar chart."""
    if not importance_dict:
        return _empty_figure("No feature importance available.")

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
        return _empty_figure("No numeric metrics available.")

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
        return _empty_figure("No common numeric metrics to compare.")

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
        return _empty_figure("Learning history is not available.")

    folds = training_history.get("folds")
    if not isinstance(folds, list):
        return _empty_figure("Learning history is not available.")

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


def plot_fold_metric_timeline(fold_metrics: pd.DataFrame | None) -> go.Figure:
    """Plot fold-wise metric timeline from fold_metrics/cv_results."""
    if fold_metrics is None or fold_metrics.empty:
        return _empty_figure("Fold metrics are not available.")
    if "fold" not in fold_metrics.columns:
        return _empty_figure("Fold metrics must include 'fold' column.")

    numeric_cols = [
        col
        for col in fold_metrics.columns
        if col != "fold" and pd.api.types.is_numeric_dtype(fold_metrics[col])
    ]
    if not numeric_cols:
        return _empty_figure("No numeric fold metrics to display.")

    fig = go.Figure()
    frame = fold_metrics.sort_values("fold")
    for col in numeric_cols[:6]:
        fig.add_trace(
            go.Scatter(
                x=frame["fold"],
                y=frame[col],
                mode="lines+markers",
                name=str(col),
            )
        )
    fig.update_layout(title="Fold Metric Timeline", xaxis_title="Fold", yaxis_title="Value")
    return _apply_theme(fig)


def plot_causal_smd(causal_summary: dict[str, object] | None) -> go.Figure:
    """Plot causal SMD diagnostics."""
    if not isinstance(causal_summary, dict):
        return _empty_figure("Causal diagnostics are not available.")
    raw_unweighted = causal_summary.get("smd_max_unweighted")
    raw_weighted = causal_summary.get("smd_max_weighted")
    try:
        unweighted = float(raw_unweighted)
        weighted = float(raw_weighted)
    except Exception:
        return _empty_figure("Causal SMD metrics are not available.")

    fig = go.Figure(
        data=[
            go.Bar(
                x=["SMD Max"],
                y=[unweighted],
                name="unweighted",
                marker={"color": COLORS["accent"]},
            ),
            go.Bar(
                x=["SMD Max"],
                y=[weighted],
                name="weighted",
                marker={"color": COLORS["secondary"]},
            ),
        ]
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=0.5,
        y0=0.1,
        y1=0.1,
        line={"color": "#ef4444", "dash": "dash"},
    )
    fig.update_layout(
        barmode="group",
        title="Causal Balance Diagnostics (SMD)",
        yaxis_title="Absolute SMD",
    )
    return _apply_theme(fig)


def plot_multi_artifact_comparison(
    metric_rows: list[dict[str, object]], metric_key: str = "delta_from_baseline"
) -> go.Figure:
    """Plot multi-artifact comparison against baseline."""
    if not metric_rows:
        return _empty_figure("Comparison data is not available.")
    frame = pd.DataFrame(metric_rows)
    required = {"metric", "artifact", metric_key}
    if not required.issubset(set(frame.columns)):
        return _empty_figure("Comparison data is missing required columns.")
    frame = frame.dropna(subset=[metric_key])
    if frame.empty:
        return _empty_figure("No comparable metric deltas available.")

    top = frame.sort_values(["metric", "artifact"]).head(200)
    fig = go.Figure()
    for artifact in sorted(top["artifact"].astype(str).unique()):
        artifact_df = top[top["artifact"].astype(str) == artifact]
        fig.add_trace(
            go.Bar(
                x=artifact_df["metric"],
                y=artifact_df[metric_key],
                name=artifact,
            )
        )
    fig.update_layout(
        title="Metric Delta vs Baseline",
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Delta",
    )
    return _apply_theme(fig)


def plot_feature_drilldown(
    observation_table: pd.DataFrame | None,
    feature: str | None,
    *,
    top_n: int = 20,
) -> go.Figure:
    """Plot feature distribution drilldown from observation_table."""
    if observation_table is None or observation_table.empty:
        return _empty_figure("Observation table is not available.")
    if not feature or feature not in observation_table.columns:
        return _empty_figure("Select a feature to view drilldown.")

    series = observation_table[feature]
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return _empty_figure("No numeric values for selected feature.")
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=values,
                    marker={"color": COLORS["secondary"]},
                    nbinsx=30,
                    name=feature,
                )
            ]
        )
        fig.update_layout(
            title=f"Feature Drilldown: {feature}",
            xaxis_title=feature,
            yaxis_title="Count",
        )
        return _apply_theme(fig)

    counts = series.astype(str).value_counts().head(max(1, int(top_n)))
    if counts.empty:
        return _empty_figure("No categorical values for selected feature.")
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker={"color": COLORS["accent"]},
            )
        ]
    )
    fig.update_layout(
        title=f"Feature Drilldown: {feature}",
        xaxis_title=feature,
        yaxis_title="Count",
    )
    return _apply_theme(fig)
