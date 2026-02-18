"""Public diagnostics for causal estimation quality."""

from __future__ import annotations

import numpy as np
import pandas as pd


def overlap_metric(propensity: np.ndarray, treatment: np.ndarray) -> float:
    """Compute overlap quality between treated and control propensity distributions."""
    treated = propensity[treatment == 1]
    control = propensity[treatment == 0]
    if len(treated) == 0 or len(control) == 0:
        return 0.0
    treated_overlap = float(np.mean((treated > 0.1) & (treated < 0.9)))
    control_overlap = float(np.mean((control > 0.1) & (control < 0.9)))
    return min(treated_overlap, control_overlap)


def compute_ess(weights: np.ndarray) -> float:
    """Compute effective sample size from non-negative weights."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0.0)]
    if w.size == 0:
        return 0.0
    numerator = float(np.sum(w) ** 2)
    denominator = float(np.sum(w**2))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def extreme_weight_ratio(weights: np.ndarray, quantile: float = 0.99) -> float:
    """Return ratio of samples exceeding specified weight quantile."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return 0.0
    threshold = float(np.quantile(w, quantile))
    return float(np.mean(w >= threshold))


def overlap_summary(
    propensity: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Return scalar overlap diagnostics used in reports and tuning."""
    e = np.asarray(propensity, dtype=float)
    t = np.asarray(treatment, dtype=int)
    summary = {
        "overlap_metric": overlap_metric(e, t),
        "propensity_min": float(np.min(e)) if e.size else 0.0,
        "propensity_max": float(np.max(e)) if e.size else 0.0,
        "propensity_p01": float(np.quantile(e, 0.01)) if e.size else 0.0,
        "propensity_p99": float(np.quantile(e, 0.99)) if e.size else 0.0,
        "extreme_propensity_ratio": float(np.mean((e < 0.01) | (e > 0.99))) if e.size else 0.0,
    }
    if weights is not None:
        summary["ess"] = compute_ess(np.asarray(weights, dtype=float))
        summary["extreme_weight_ratio"] = extreme_weight_ratio(np.asarray(weights, dtype=float))
    return summary


def _weighted_mean_var(values: np.ndarray, weights: np.ndarray | None) -> tuple[float, float]:
    if weights is None:
        mean = float(np.mean(values))
        var = float(np.var(values))
        return mean, var
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0, 0.0
    mean = float(np.average(values, weights=weights))
    var = float(np.average((values - mean) ** 2, weights=weights))
    return mean, var


def max_standardized_mean_difference(
    covariates: pd.DataFrame,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Compute max SMD across covariate columns."""
    if covariates.empty:
        return 0.0
    t_mask = treatment == 1
    c_mask = treatment == 0
    if not np.any(t_mask) or not np.any(c_mask):
        return 0.0

    smd_values: list[float] = []
    for col in covariates.columns:
        values = pd.to_numeric(covariates[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(values).any():
            continue
        w_t = weights[t_mask] if weights is not None else None
        w_c = weights[c_mask] if weights is not None else None
        mean_t, var_t = _weighted_mean_var(values[t_mask], w_t)
        mean_c, var_c = _weighted_mean_var(values[c_mask], w_c)
        pooled_sd = float(np.sqrt(max((var_t + var_c) / 2.0, 0.0)))
        if pooled_sd <= 1e-12:
            smd_values.append(0.0)
            continue
        smd_values.append(abs(mean_t - mean_c) / pooled_sd)

    if not smd_values:
        return 0.0
    return float(np.max(smd_values))
