from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.diagnostics.causal_diag import (
    compute_balance_smd,
    compute_overlap_stats,
    compute_trimming_comparison,
    get_if_outliers,
)


def test_compute_overlap_stats_contract() -> None:
    stats = compute_overlap_stats([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert {"min", "max", "p01", "p99", "extreme_ratio"} <= set(stats)


def test_compute_balance_smd_returns_frame() -> None:
    cov = pd.DataFrame({"x1": [1.0, 2.0, 1.5, 3.0], "x2": [0.2, 0.1, 0.3, 0.4]})
    smd = compute_balance_smd(cov, [0, 1, 0, 1])
    assert {"feature", "smd"} <= set(smd.columns)


def test_compute_trimming_comparison_and_if_outliers() -> None:
    table = pd.DataFrame({"weight": [1.0, 2.0, 3.0, 10.0], "psi": [0.1, 0.2, 0.3, 0.4]})

    def _estimate_fn(df: pd.DataFrame) -> float:
        return float(np.average(df["psi"], weights=df["weight"]))

    trim = compute_trimming_comparison(_estimate_fn, table, trim_levels=[0.01, 0.05])
    assert {"trim_level", "estimate", "ess", "extreme_weight_ratio"} <= set(trim.columns)

    outliers = get_if_outliers([0.1, 0.2, 2.5, 0.3], pd.DataFrame({"x": [1, 2, 3, 4]}), 75)
    assert not outliers.empty
