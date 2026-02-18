"""Causal diagnostics helpers for notebooks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from veldra.causal.diagnostics import compute_ess, extreme_weight_ratio


def _ensure_parent(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_propensity_distribution(propensity, treatment, save_path) -> None:
    path = _ensure_parent(save_path)
    propensity = np.asarray(propensity, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(propensity[treatment == 1], bins=30, alpha=0.6, label="treated")
    ax.hist(propensity[treatment == 0], bins=30, alpha=0.6, label="control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weight_distribution(weights, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.asarray(weights, dtype=float), bins=30, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compute_overlap_stats(propensity, treatment) -> dict[str, float]:
    e = np.asarray(propensity, dtype=float)
    return {
        "min": float(np.min(e)) if len(e) else 0.0,
        "max": float(np.max(e)) if len(e) else 0.0,
        "p01": float(np.quantile(e, 0.01)) if len(e) else 0.0,
        "p99": float(np.quantile(e, 0.99)) if len(e) else 0.0,
        "extreme_ratio": float(np.mean((e < 0.01) | (e > 0.99))) if len(e) else 0.0,
    }


def compute_balance_smd(covariates, treatment, weights=None) -> pd.DataFrame:
    x = pd.DataFrame(covariates)
    t = np.asarray(treatment, dtype=int)
    w = None if weights is None else np.asarray(weights, dtype=float)

    records: list[dict[str, float | str]] = []
    for col in x.columns:
        values = pd.to_numeric(x[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(values).any():
            continue
        t_vals = values[t == 1]
        c_vals = values[t == 0]
        if len(t_vals) == 0 or len(c_vals) == 0:
            continue
        if w is None:
            mean_t = float(np.mean(t_vals))
            mean_c = float(np.mean(c_vals))
            var_t = float(np.var(t_vals))
            var_c = float(np.var(c_vals))
        else:
            w_t = w[t == 1]
            w_c = w[t == 0]
            mean_t = float(np.average(t_vals, weights=w_t))
            mean_c = float(np.average(c_vals, weights=w_c))
            var_t = float(np.average((t_vals - mean_t) ** 2, weights=w_t))
            var_c = float(np.average((c_vals - mean_c) ** 2, weights=w_c))
        pooled = float(np.sqrt(max((var_t + var_c) / 2.0, 0.0)))
        smd = 0.0 if pooled <= 1e-12 else abs(mean_t - mean_c) / pooled
        records.append({"feature": str(col), "smd": float(smd)})

    return pd.DataFrame.from_records(records)


def plot_love_plot(smd_unweighted, smd_weighted, save_path) -> None:
    path = _ensure_parent(save_path)
    left = pd.DataFrame(smd_unweighted).set_index("feature")
    right = pd.DataFrame(smd_weighted).set_index("feature")
    merged = left.join(right, lsuffix="_unweighted", rsuffix="_weighted", how="outer").fillna(0.0)
    merged = merged.sort_values("smd_unweighted", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, int(0.3 * len(merged)))))
    y = np.arange(len(merged))
    ax.scatter(merged["smd_unweighted"], y, label="unweighted")
    ax.scatter(merged["smd_weighted"], y, label="weighted")
    ax.axvline(0.1, color="r", linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(merged.index)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compute_trimming_comparison(
    estimate_fn: Callable[[pd.DataFrame], float],
    observation_table: pd.DataFrame,
    trim_levels: list[float] | tuple[float, ...] = (0.01, 0.05),
) -> pd.DataFrame:
    frame = pd.DataFrame(observation_table).copy()
    if "weight" not in frame.columns:
        return pd.DataFrame(columns=["trim_level", "estimate", "ess", "extreme_weight_ratio"])

    records: list[dict[str, float]] = []
    weights = np.asarray(frame["weight"], dtype=float)
    for level in trim_levels:
        low = float(np.quantile(weights, level))
        high = float(np.quantile(weights, 1.0 - level))
        clipped = frame.copy()
        clipped["weight"] = np.clip(weights, low, high)
        clipped_estimate = float(estimate_fn(clipped))
        records.append(
            {
                "trim_level": float(level),
                "estimate": clipped_estimate,
                "ess": float(compute_ess(clipped["weight"].to_numpy(dtype=float))),
                "extreme_weight_ratio": float(
                    extreme_weight_ratio(clipped["weight"].to_numpy(dtype=float))
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_if_distribution(if_values, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.asarray(if_values, dtype=float), bins=40, alpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def get_if_outliers(if_values, observation_table, percentile: float = 99) -> pd.DataFrame:
    values = np.asarray(if_values, dtype=float)
    threshold = float(np.quantile(np.abs(values), percentile / 100.0))
    frame = pd.DataFrame(observation_table).copy()
    frame["if_value"] = values
    return frame.loc[np.abs(frame["if_value"]) >= threshold].copy()


def plot_parallel_trends(means_treated, means_control, time_labels, save_path) -> None:
    path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_labels, means_treated, marker="o", label="treated")
    ax.plot(time_labels, means_control, marker="o", label="control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
