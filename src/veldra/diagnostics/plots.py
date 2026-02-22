"""Plotting helpers for diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

_EMPTY_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf5\x17\xd4\x8f"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _ensure_parent(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_plotly_or_empty(fig: go.Figure, save_path: str | Path) -> Path:
    path = _ensure_parent(save_path)
    try:
        fig.write_image(path)
    except Exception:
        path.write_bytes(_EMPTY_PNG_1X1)
    return path


def _write_empty_png(save_path: str | Path) -> None:
    path = _ensure_parent(save_path)
    path.write_bytes(_EMPTY_PNG_1X1)


def plot_error_histogram(
    residuals_in, residuals_out, metrics_in, metrics_out, save_path
) -> go.Figure:  # noqa: ARG001
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=np.asarray(residuals_in, dtype=float), name="in", opacity=0.65, nbinsx=30)
    )
    fig.add_trace(
        go.Histogram(x=np.asarray(residuals_out, dtype=float), name="out", opacity=0.65, nbinsx=30)
    )
    fig.update_layout(
        title="Residual Distribution",
        barmode="overlay",
        xaxis_title="Residual",
        yaxis_title="Count",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_roc_comparison(y_true_in, y_score_in, y_true_out, y_score_out, save_path) -> go.Figure:
    fpr_in, tpr_in, _ = roc_curve(y_true_in, y_score_in)
    fpr_out, tpr_out, _ = roc_curve(y_true_out, y_score_out)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_in,
            y=tpr_in,
            mode="lines",
            name="in",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fpr_out,
            y=tpr_out,
            mode="lines",
            name="out",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="baseline",
            line={"dash": "dash", "color": "#666666"},
        )
    )
    fig.update_layout(
        title="ROC Comparison",
        xaxis_title="FPR",
        yaxis_title="TPR",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path,
) -> go.Figure:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    class_labels = [str(v) for v in class_names]
    labels = (
        list(range(len(class_labels)))
        if np.issubdtype(y_true_arr.dtype, np.number)
        else class_labels
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_labels,
            y=class_labels,
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            hovertemplate="pred=%{x}<br>true=%{y}<br>count=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_roc_multiclass(y_true, y_proba, class_names, save_path) -> go.Figure:
    y_true_arr = np.asarray(y_true, dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    labels = [str(v) for v in class_names]
    fig = go.Figure()
    for class_idx, class_label in enumerate(labels):
        y_bin = (y_true_arr == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, class_idx])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{class_label} vs rest",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="baseline",
            line={"dash": "dash", "color": "#666666"},
        )
    )
    fig.update_layout(
        title="Multiclass ROC (OvR)",
        xaxis_title="FPR",
        yaxis_title="TPR",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_lift_chart(y_true, y_score, save_path) -> None:
    path = _ensure_parent(save_path)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    population = np.arange(1, len(y_sorted) + 1)
    baseline = population * (np.mean(y_true) if len(y_true) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(population, cum_pos, label="model")
    ax.plot(population, baseline, "--", label="baseline")
    ax.set_title("Lift Chart")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_nll_histogram(nll_in, nll_out, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nll_in, bins=30, alpha=0.6, label="in")
    ax.hist(nll_out, bins=30, alpha=0.6, label="out")
    ax.set_title("Negative Log-Likelihood")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_true_class_prob_histogram(prob_in, prob_out, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prob_in, bins=30, alpha=0.6, label="in")
    ax.hist(prob_out, bins=30, alpha=0.6, label="out")
    ax.set_title("P(true class)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeseries_prediction(time_index, y_true, y_pred, split_point, save_path) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=y_true, mode="lines", name="actual"))
    fig.add_trace(go.Scatter(x=time_index, y=y_pred, mode="lines", name="pred"))
    fig.add_vline(x=split_point, line_dash="dash", line_color="#444444")
    fig.update_layout(
        title="Timeseries Prediction",
        xaxis_title="time",
        yaxis_title="value",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_timeseries_residual(time_index, residuals, split_point, save_path) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=residuals, mode="lines", name="residual"))
    fig.add_hline(y=0.0, line_dash="dash", line_color="#666666")
    fig.add_vline(x=split_point, line_dash="dash", line_color="#444444")
    fig.update_layout(
        title="Timeseries Residual",
        xaxis_title="time",
        yaxis_title="residual",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_learning_curve(training_history: dict | None, save_path: str | Path) -> go.Figure:
    if not isinstance(training_history, dict):
        _write_empty_png(save_path)
        return go.Figure()
    folds = training_history.get("folds")
    if not isinstance(folds, list) or len(folds) == 0:
        _write_empty_png(save_path)
        return go.Figure()

    fold_series: list[np.ndarray] = []
    fig = go.Figure()
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        eval_history = fold.get("eval_history")
        if not isinstance(eval_history, dict) or len(eval_history) == 0:
            continue
        first_metric = next(iter(eval_history.values()))
        if not isinstance(first_metric, (list, tuple, np.ndarray)):
            continue
        values = np.asarray(first_metric, dtype=float)
        if values.ndim != 1 or values.size == 0:
            continue
        fold_series.append(values)
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(values) + 1),
                y=values,
                mode="lines",
                line={"width": 1},
                opacity=0.4,
                name=f"fold_{fold.get('fold', len(fold_series))}",
            )
        )

    if len(fold_series) == 0:
        _write_empty_png(save_path)
        return go.Figure()

    max_len = max(len(v) for v in fold_series)
    stacked = np.full((len(fold_series), max_len), np.nan, dtype=float)
    for idx, values in enumerate(fold_series):
        stacked[idx, : len(values)] = values
    mean_curve = np.nanmean(stacked, axis=0)
    valid = ~np.isnan(mean_curve)
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, max_len + 1)[valid],
            y=mean_curve[valid],
            mode="lines",
            line={"width": 3},
            name="mean",
        )
    )
    fig.update_layout(
        title="Learning Curve (CV Folds)",
        xaxis_title="iteration",
        yaxis_title="metric",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_pinball_histogram(pinball_in, pinball_out, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pinball_in, bins=30, alpha=0.6, label="in")
    ax.hist(pinball_out, bins=30, alpha=0.6, label="out")
    ax.set_title("Pinball Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_frontier_scatter(y_true, y_pred, save_path) -> None:
    path = _ensure_parent(save_path)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred, y_true, alpha=0.6)
    low = float(min(np.min(y_true), np.min(y_pred)))
    high = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([low, high], [low, high], "k--", alpha=0.6)
    ax.set_xlabel("pred")
    ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(
    importance_df: pd.DataFrame, importance_type: str, save_path
) -> go.Figure:
    frame = importance_df.head(20).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=frame["importance"],
            y=frame["feature"],
            orientation="h",
            marker={"color": "#1f77b4"},
        )
    )
    fig.update_layout(
        title=f"Feature Importance ({importance_type})",
        xaxis_title="importance",
        yaxis_title="feature",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig


def plot_shap_summary(shap_df: pd.DataFrame, X: pd.DataFrame, save_path) -> go.Figure:  # noqa: ARG001
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(20)
    fig = go.Figure(
        go.Bar(
            x=mean_abs.values[::-1],
            y=mean_abs.index[::-1],
            orientation="h",
            marker={"color": "#2ca02c"},
        )
    )
    fig.update_layout(
        title="SHAP Summary (mean |contrib|)",
        xaxis_title="mean |contrib|",
        yaxis_title="feature",
        template="plotly_white",
    )
    _save_plotly_or_empty(fig, save_path)
    return fig
