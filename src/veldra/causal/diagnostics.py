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
