from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.causal.diagnostics import (
    compute_ess,
    extreme_weight_ratio,
    max_standardized_mean_difference,
    overlap_summary,
)


def test_compute_ess_handles_empty_zero_and_positive_weights() -> None:
    assert compute_ess(np.array([])) == 0.0
    assert compute_ess(np.array([0.0, 0.0])) == 0.0
    ess = compute_ess(np.array([1.0, 2.0, 3.0]))
    assert ess == (6.0**2) / 14.0


def test_extreme_weight_ratio_handles_empty_and_quantile_boundary() -> None:
    assert extreme_weight_ratio(np.array([]), quantile=0.99) == 0.0
    ratio = extreme_weight_ratio(np.array([1.0, 2.0, 3.0, 4.0]), quantile=0.5)
    assert ratio == 0.5


def test_overlap_summary_handles_missing_group_and_weight_summary() -> None:
    treated_only = overlap_summary(
        np.array([0.2, 0.4, 0.6], dtype=float),
        np.array([1, 1, 1], dtype=int),
        weights=np.array([1.0, 2.0, 3.0], dtype=float),
    )
    assert treated_only["overlap_metric"] == 0.0
    assert treated_only["propensity_min"] == 0.2
    assert treated_only["propensity_max"] == 0.6
    assert treated_only["ess"] > 0.0
    assert 0.0 <= treated_only["extreme_weight_ratio"] <= 1.0


def test_max_smd_skips_nan_columns_and_handles_one_sided_treatment() -> None:
    covariates = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.9, 1.0],
            "x2": [1.0, np.nan, 2.0, np.nan],
        }
    )
    treatment = np.array([0, 0, 1, 1], dtype=int)
    value = max_standardized_mean_difference(covariates, treatment)
    assert value >= 0.0

    one_sided = max_standardized_mean_difference(
        pd.DataFrame({"x1": [0.1, 0.2, 0.3]}),
        np.array([1, 1, 1], dtype=int),
    )
    assert one_sided == 0.0
