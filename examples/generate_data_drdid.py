"""Generate synthetic datasets for DR-DiD validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_panel_data(
    n_units: int = 3000,
    *,
    seed: int = 42,
    true_att: float = 2000.0,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 58, size=n_units)
    education = rng.integers(8, 20, size=n_units)
    baseline_skill = rng.normal(0.0, 1.0, size=n_units)

    logits = -1.3 + 0.03 * (age - 30) + 0.15 * (education - 12) + 0.7 * baseline_skill
    propensity = _sigmoid(logits)
    treatment = rng.binomial(1, propensity, size=n_units)

    baseline_income = (
        12000.0
        + 350.0 * (age - 30)
        + 500.0 * (education - 12)
        + 2000.0 * baseline_skill
        + rng.normal(0.0, 2500.0, size=n_units)
    )
    macro_trend = 800.0 + 180.0 * baseline_skill
    pre_income = baseline_income + rng.normal(0.0, 1200.0, size=n_units)
    post_income = (
        baseline_income
        + macro_trend
        + treatment * true_att
        + rng.normal(0.0, 1200.0, size=n_units)
    )

    pre = pd.DataFrame(
        {
            "unit_id": np.arange(n_units),
            "time": 0,
            "post": 0,
            "treatment": treatment,
            "age": age,
            "education": education,
            "baseline_skill": baseline_skill,
            "outcome": pre_income,
            "true_att": true_att,
        }
    )
    post = pd.DataFrame(
        {
            "unit_id": np.arange(n_units),
            "time": 1,
            "post": 1,
            "treatment": treatment,
            "age": age,
            "education": education,
            "baseline_skill": baseline_skill,
            "outcome": post_income,
            "true_att": true_att,
        }
    )
    return pd.concat([pre, post], ignore_index=True), true_att


def generate_repeated_cs_data(
    n_pre: int = 2500,
    n_post: int = 2500,
    *,
    seed: int = 43,
    true_att: float = 1800.0,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    n_total = n_pre + n_post
    post = np.concatenate([np.zeros(n_pre, dtype=int), np.ones(n_post, dtype=int)])

    age = rng.integers(18, 58, size=n_total)
    education = rng.integers(8, 20, size=n_total)
    baseline_skill = rng.normal(0.0, 1.0, size=n_total)

    logits = (
        -1.1
        + 0.04 * (age - 30)
        + 0.13 * (education - 12)
        + 0.65 * baseline_skill
        + 0.10 * post
    )
    propensity = _sigmoid(logits)
    treatment = rng.binomial(1, propensity, size=n_total)

    base = (
        11000.0
        + 300.0 * (age - 30)
        + 450.0 * (education - 12)
        + 1800.0 * baseline_skill
    )
    outcome = (
        base
        + 700.0 * post
        + true_att * treatment * post
        + rng.normal(0.0, 2600.0, size=n_total)
    )
    df = pd.DataFrame(
        {
            "time": post,
            "post": post,
            "treatment": treatment,
            "age": age,
            "education": education,
            "baseline_skill": baseline_skill,
            "outcome": outcome,
            "true_att": true_att,
        }
    )
    return df, true_att


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DR-DiD synthetic datasets.")
    parser.add_argument("--n-units", type=int, default=3000)
    parser.add_argument("--n-pre", type=int, default=2500)
    parser.add_argument("--n-post", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples") / "data",
        help="Output directory for generated CSV files.",
    )
    args = parser.parse_args()

    panel_df, panel_tau = generate_panel_data(n_units=args.n_units, seed=args.seed)
    repeated_df, repeated_tau = generate_repeated_cs_data(
        n_pre=args.n_pre,
        n_post=args.n_post,
        seed=args.seed + 1,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.out_dir / "drdid_panel.csv"
    repeated_path = args.out_dir / "drdid_repeated_cs.csv"
    summary_path = args.out_dir / "drdid_summary.json"
    panel_df.to_csv(panel_path, index=False)
    repeated_df.to_csv(repeated_path, index=False)

    summary = {
        "panel_path": str(panel_path),
        "repeated_cs_path": str(repeated_path),
        "panel_rows": int(len(panel_df)),
        "repeated_cs_rows": int(len(repeated_df)),
        "panel_true_att": float(panel_tau),
        "repeated_cs_true_att": float(repeated_tau),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

