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


def test_causal_diag_plot_functions_write_files(tmp_path) -> None:
    from veldra.diagnostics.causal_diag import (
        plot_if_distribution,
        plot_love_plot,
        plot_parallel_trends,
        plot_propensity_distribution,
        plot_weight_distribution,
    )

    propensity = [0.1, 0.2, 0.7, 0.9]
    treatment = [0, 0, 1, 1]
    weights = [1.0, 2.0, 3.0, 4.0]

    p1 = tmp_path / "plots" / "propensity.png"
    p2 = tmp_path / "plots" / "weights.png"
    p3 = tmp_path / "plots" / "love.png"
    p4 = tmp_path / "plots" / "if_hist.png"
    p5 = tmp_path / "plots" / "parallel.png"

    plot_propensity_distribution(propensity, treatment, p1)
    plot_weight_distribution(weights, p2)
    plot_love_plot(
        [{"feature": "x1", "smd": 0.2}],
        [{"feature": "x1", "smd": 0.1}],
        p3,
    )
    plot_if_distribution([0.1, -0.2, 0.3], p4)
    plot_parallel_trends([1.0, 1.2], [0.8, 0.9], ["t0", "t1"], p5)

    assert p1.is_file()
    assert p2.is_file()
    assert p3.is_file()
    assert p4.is_file()
    assert p5.is_file()


def test_compute_balance_smd_skips_nan_and_missing_group_and_handles_weights() -> None:
    cov = pd.DataFrame(
        {
            "nan_col": [1.0, np.nan, 3.0, 4.0],
            "ok_col": [1.0, 2.0, 1.2, 2.2],
            "only_treated": [5.0, 6.0, 7.0, 8.0],
        }
    )

    smd = compute_balance_smd(cov, [0, 1, 0, 1], weights=[1.0, 2.0, 1.0, 2.0])
    assert set(smd["feature"]) == {"ok_col", "only_treated"}

    treated_only = compute_balance_smd(pd.DataFrame({"x": [1.0, 2.0]}), [1, 1])
    assert treated_only.empty


def test_compute_trimming_comparison_without_weight_column_returns_empty() -> None:
    out = compute_trimming_comparison(lambda _df: 0.0, pd.DataFrame({"psi": [1.0, 2.0]}))
    assert list(out.columns) == ["trim_level", "estimate", "ess", "extreme_weight_ratio"]
    assert out.empty
