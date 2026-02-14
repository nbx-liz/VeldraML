from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.causal import max_standardized_mean_difference, overlap_metric


def test_overlap_metric_edge_cases() -> None:
    e_hat = np.array([0.2, 0.4, 0.8], dtype=float)
    treated_only = np.array([1, 1, 1], dtype=int)
    assert overlap_metric(e_hat, treated_only) == 0.0

    treatment = np.array([0, 0, 1, 1], dtype=int)
    e_hat2 = np.array([0.05, 0.15, 0.85, 0.95], dtype=float)
    value = overlap_metric(e_hat2, treatment)
    assert 0.0 <= value <= 1.0


def test_max_standardized_mean_difference_weighted_and_empty() -> None:
    cov = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "x2": [1.0, 1.1, 2.0, 2.1]})
    treatment = np.array([0, 0, 1, 1], dtype=int)
    weights = np.array([1.0, 1.0, 1.2, 1.3], dtype=float)
    unweighted = max_standardized_mean_difference(cov, treatment)
    weighted = max_standardized_mean_difference(cov, treatment, weights=weights)
    assert unweighted >= 0.0
    assert weighted >= 0.0
    assert max_standardized_mean_difference(pd.DataFrame(), treatment) == 0.0
