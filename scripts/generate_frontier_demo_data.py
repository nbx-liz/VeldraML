"""Generate reproducible frontier demo data for Phase35.0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("data/demo/frontier")
DEFAULT_SEED = 42

PARAMS: dict[str, Any] = {
    "sigma_hq": 0.20,
    "sigma_dept": 0.15,
    "sigma_u": 0.30,
    "w": [0.6, 0.8, 1.0, 1.2, 1.4],
    "m": [1.2, 1.6, 2.0],
    "span0": 8.0,
    "b_span": 0.25,
    "a_L": 0.15,
    "sigma_touch": 0.15,
    "sigma_opp": 0.20,
    "sigma_meet": 0.20,
    "sigma_prop": 0.20,
    "sigma_win": 0.20,
    "sigma_price": 0.25,
    "sigma_sales": 0.25,
    "sigma_D": 0.40,
    "phi_opp": 0.50,
    "phi_meet": 0.50,
    "phi_prop": 0.50,
    "phi_win": 0.50,
    "omega3": 0.15,
    "delta3": 0.15,
}

HC_COLS = [f"hc_g{i}" for i in range(1, 11)]
OUTPUT_COLS = [
    "hq_id",
    "dept_id",
    "sec_id",
    "month",
    *HC_COLS,
    "territory_potential",
    "discount_amount_total",
    "touches",
    "opps_created",
    "meetings_held",
    "proposals_sent",
    "net_sales",
    "u_s",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def build_org_master(seed: int) -> pd.DataFrame:
    """Build organization master data (5 HQ x 50 DEPT x 500 SEC)."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str, str]] = []
    for h_idx in range(1, 6):
        hq_id = f"HQ{h_idx:02d}"
        for d_offset in range(1, 11):
            dept_global = (h_idx - 1) * 10 + d_offset
            dept_id = f"DEPT{dept_global:02d}"
            for s_offset in range(1, 11):
                sec_global = (dept_global - 1) * 10 + s_offset
                sec_id = f"SEC{sec_global:03d}"
                rows.append((hq_id, dept_id, sec_id))

    master = pd.DataFrame(rows, columns=["hq_id", "dept_id", "sec_id"])
    hq_effect = {
        f"HQ{i:02d}": value
        for i, value in enumerate(rng.normal(0.0, PARAMS["sigma_hq"], size=5), start=1)
    }
    dept_effect = {
        f"DEPT{i:02d}": value
        for i, value in enumerate(rng.normal(0.0, PARAMS["sigma_dept"], size=50), start=1)
    }
    master["_alpha_h"] = master["hq_id"].map(hq_effect).astype(float)
    master["_alpha_d"] = master["dept_id"].map(dept_effect).astype(float)
    master["u_s"] = np.abs(rng.normal(0.0, PARAMS["sigma_u"], size=len(master)))
    master["territory_potential"] = rng.lognormal(mean=1.0, sigma=0.35, size=len(master))

    hc_lambdas = [6.0, 5.0, 4.0, 3.0, 2.0, 2.0, 1.5, 1.2, 1.0, 0.8]
    for col, lam in zip(HC_COLS, hc_lambdas, strict=True):
        master[col] = rng.poisson(lam=lam, size=len(master))

    w = np.asarray(PARAMS["w"], dtype=float)
    m = np.asarray(PARAMS["m"], dtype=float)
    c_ic = (master[[f"hc_g{i}" for i in range(1, 6)]].to_numpy() * w).sum(axis=1)
    c_m = (master[[f"hc_g{i}" for i in range(6, 9)]].to_numpy() * m).sum(axis=1)
    lead_factor = 1.0 + PARAMS["a_L"] * np.log1p(c_m)
    span = master[[f"hc_g{i}" for i in range(1, 6)]].sum(axis=1).to_numpy() / np.maximum(
        1,
        master[[f"hc_g{i}" for i in range(6, 9)]].sum(axis=1).to_numpy(),
    )
    span_penalty = np.exp(
        -PARAMS["b_span"] * (np.log(np.maximum(span, 1e-9)) - np.log(PARAMS["span0"])) ** 2
    )
    master["_cap"] = c_ic * lead_factor * span_penalty * np.exp(
        master["_alpha_h"] + master["_alpha_d"]
    )
    return master


def simulate_monthly(
    master: pd.DataFrame,
    months: list[str],
    month_start_index: int,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate monthly frontier observations with fixed section-level effects."""
    cap = master["_cap"].to_numpy(dtype=float)
    tau = np.log(master["territory_potential"].to_numpy(dtype=float))
    u_s = master["u_s"].to_numpy(dtype=float)

    chunks: list[pd.DataFrame] = []
    for offset, month in enumerate(months):
        t = month_start_index + offset
        season = 0.10 * np.sin(2.0 * np.pi * t / 12.0) + 0.05 * np.cos(2.0 * np.pi * t / 12.0)

        eps_d = rng.normal(0.0, params["sigma_D"], size=len(master))
        log_discount = (
            1.2
            + 0.35 * np.log1p(cap)
            + 0.65 * tau
            + 0.35 * season
            + eps_d
        )
        discount = np.exp(log_discount)
        scale_d = float(np.median(discount))

        v_touch = rng.normal(0.0, params["sigma_touch"], size=len(master))
        lam_touch = np.exp(
            1.1 + 0.55 * np.log1p(cap) + 0.45 * tau + 0.35 * season + v_touch
        )
        touches = rng.poisson(np.clip(lam_touch, 1e-6, None))

        v_opp = rng.normal(0.0, params["sigma_opp"], size=len(master))
        p_opp = _sigmoid(
            -0.4
            + 0.22 * np.log1p(cap)
            + 0.20 * tau
            + 0.15 * season
            - params["phi_opp"] * u_s
            + v_opp
        )
        opps = rng.binomial(touches, np.clip(p_opp, 1e-6, 1.0 - 1e-6))

        v_meet = rng.normal(0.0, params["sigma_meet"], size=len(master))
        p_meet = _sigmoid(
            -0.2
            + 0.18 * np.log1p(cap)
            + 0.16 * tau
            + 0.10 * season
            - params["phi_meet"] * u_s
            + v_meet
        )
        meetings = rng.binomial(opps, np.clip(p_meet, 1e-6, 1.0 - 1e-6))

        v_prop = rng.normal(0.0, params["sigma_prop"], size=len(master))
        p_prop = _sigmoid(
            0.0
            + 0.15 * np.log1p(cap)
            + 0.14 * tau
            + 0.08 * season
            - params["phi_prop"] * u_s
            + v_prop
        )
        proposals = rng.binomial(meetings, np.clip(p_prop, 1e-6, 1.0 - 1e-6))

        v_win = rng.normal(0.0, params["sigma_win"], size=len(master))
        p_win = _sigmoid(
            -0.2
            + 0.20 * np.log1p(cap)
            + 0.15 * tau
            + params["omega3"] * np.log1p(discount / np.maximum(scale_d, 1e-6))
            - params["phi_win"] * u_s
            + v_win
        )
        won = rng.binomial(proposals, np.clip(p_win, 1e-6, 1.0 - 1e-6))

        v_price = rng.normal(0.0, params["sigma_price"], size=len(master))
        log_price = (
            8.7
            + 0.35 * tau
            + 0.10 * season
            - params["delta3"] * np.log1p(discount / np.maximum(scale_d, 1e-6))
            + v_price
        )

        latent_sales = won * np.exp(log_price)
        v_sales = rng.normal(0.0, params["sigma_sales"], size=len(master))
        net_sales = latent_sales * np.exp(v_sales - u_s)

        frame = master[
            ["hq_id", "dept_id", "sec_id", *HC_COLS, "territory_potential", "u_s"]
        ].copy()
        frame["month"] = month
        frame["discount_amount_total"] = discount
        frame["touches"] = touches
        frame["opps_created"] = opps
        frame["meetings_held"] = meetings
        frame["proposals_sent"] = proposals
        frame["net_sales"] = net_sales
        chunks.append(frame[OUTPUT_COLS])

    return pd.concat(chunks, ignore_index=True)


def generate_frontier_demo_data(
    out_dir: str | Path,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train/eval and latest frontier datasets and persist both CSV files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    master = build_org_master(seed=seed)

    train_months = pd.period_range("2023-01", "2024-12", freq="M").astype(str).tolist()
    latest_months = pd.period_range("2025-01", "2025-12", freq="M").astype(str).tolist()

    train_eval = simulate_monthly(master, train_months, month_start_index=1, params=PARAMS, rng=rng)
    latest = simulate_monthly(master, latest_months, month_start_index=25, params=PARAMS, rng=rng)

    train_eval.to_csv(out_path / "train_eval.csv", index=False)
    latest.to_csv(out_path / "latest.csv", index=False)
    return train_eval, latest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: data/demo/frontier).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_eval, latest = generate_frontier_demo_data(out_dir=args.out_dir, seed=args.seed)
    print(f"Saved: {args.out_dir / 'train_eval.csv'} (rows={len(train_eval)})")
    print(f"Saved: {args.out_dir / 'latest.csv'} (rows={len(latest)})")
    print(f"train_eval net_sales>0 ratio: {(train_eval['net_sales'] > 0).mean():.4f}")
    print(f"latest net_sales>0 ratio: {(latest['net_sales'] > 0).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
