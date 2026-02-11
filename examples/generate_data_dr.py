"""Generate synthetic DR validation datasets with confounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_frame(n_samples: int, random_state: int, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    x1 = rng.normal(loc=0.2 if drift else 0.0, scale=1.0, size=n_samples)
    x2 = rng.normal(loc=-0.3 if drift else 0.0, scale=1.2, size=n_samples)
    x3 = rng.uniform(-1.0, 1.0, size=n_samples)

    # Heterogeneous treatment effect (ground truth).
    true_tau = 1.4 + 0.7 * x1 - 0.25 * x2 + 0.15 * x3

    # Confounded treatment assignment.
    logit_p = -0.1 + 0.9 * x1 - 0.6 * x2 + 0.4 * x3
    p_treat = np.clip(_sigmoid(logit_p), 0.02, 0.98)
    treatment = rng.binomial(1, p_treat, size=n_samples)

    # Baseline outcome + treatment uplift + noise.
    baseline = 3.0 + 1.1 * x1 + 0.6 * x2 - 0.4 * x3
    noise = rng.normal(0.0, 0.8 if drift else 0.6, size=n_samples)
    outcome = baseline + treatment * true_tau + noise

    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(float),
            "true_tau": true_tau.astype(float),
            "p_treat_true": p_treat.astype(float),
        }
    )


def generate_dr_datasets(
    n_train: int = 5000,
    n_test: int = 2000,
    seed: int = 42,
    out_dir: str | Path = Path("examples") / "data",
) -> dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df = _generate_frame(n_train, random_state=seed, drift=False)
    test_df = _generate_frame(n_test, random_state=seed + 1, drift=True)

    train_csv = out_path / "dr_train.csv"
    test_csv = out_path / "dr_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    stats = {
        "train_path": str(train_csv),
        "test_path": str(test_csv),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "true_ate_train": float(train_df["true_tau"].mean()),
        "true_att_train": float(train_df.loc[train_df["treatment"] == 1, "true_tau"].mean()),
        "treated_rate_train": float(train_df["treatment"].mean()),
    }
    (out_path / "dr_generation_summary.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DR validation datasets.")
    parser.add_argument("--n-train", type=int, default=5000, help="Number of training samples.")
    parser.add_argument("--n-test", type=int, default=2000, help="Number of test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("examples") / "data"),
        help="Output directory for generated CSV files.",
    )
    args = parser.parse_args()
    stats = generate_dr_datasets(
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

