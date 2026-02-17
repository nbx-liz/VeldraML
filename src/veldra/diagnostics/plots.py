"""Matplotlib plotting helpers for diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def _ensure_parent(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_error_histogram(residuals_in, residuals_out, metrics_in, metrics_out, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals_in, bins=30, alpha=0.6, label="in")
    ax.hist(residuals_out, bins=30, alpha=0.6, label="out")
    ax.set_title("Residual Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc_comparison(y_true_in, y_score_in, y_true_out, y_score_out, save_path) -> None:
    path = _ensure_parent(save_path)
    fpr_in, tpr_in, _ = roc_curve(y_true_in, y_score_in)
    fpr_out, tpr_out, _ = roc_curve(y_true_out, y_score_out)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_in, tpr_in, label="in")
    ax.plot(fpr_out, tpr_out, label="out")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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


def plot_timeseries_prediction(time_index, y_true, y_pred, split_point, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_index, y_true, label="actual")
    ax.plot(time_index, y_pred, label="pred")
    ax.axvline(split_point, color="k", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeseries_residual(time_index, residuals, split_point, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_index, residuals, label="residual")
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(split_point, color="k", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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


def plot_feature_importance(importance_df: pd.DataFrame, importance_type: str, save_path) -> None:
    path = _ensure_parent(save_path)
    frame = importance_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(frame["feature"], frame["importance"])
    ax.set_title(f"Feature Importance ({importance_type})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_shap_summary(shap_df: pd.DataFrame, X: pd.DataFrame, save_path) -> None:
    path = _ensure_parent(save_path)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(mean_abs.index[::-1], mean_abs.values[::-1])
    ax.set_title("SHAP Summary (mean |contrib|)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
