"""Prepare synthetic frontier demo CSV (network-free)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - import path depends on launch style
    from examples.common import DEFAULT_FRONTIER_DATA_PATH, DEFAULT_TARGET, format_error
except ModuleNotFoundError:  # pragma: no cover
    from common import DEFAULT_FRONTIER_DATA_PATH, DEFAULT_TARGET, format_error


def build_frontier_frame(rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with asymmetric positive noise for quantile frontier demos."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.5, 2.5, size=rows)
    x2 = rng.normal(loc=0.0, scale=1.0, size=rows)
    base = 2.2 + 1.8 * x1 - 0.7 * x2
    noise = rng.normal(loc=0.0, scale=0.25 + 0.35 * np.abs(x1), size=rows)
    tail = rng.exponential(scale=0.35, size=rows)
    y = base + noise + tail
    return pd.DataFrame({"x1": x1, "x2": x2, DEFAULT_TARGET: y})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_FRONTIER_DATA_PATH),
        help="Output CSV path (default: examples/data/frontier_demo.csv).",
    )
    parser.add_argument("--rows", type=int, default=800, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out_path)

    try:
        frame = build_frontier_frame(rows=args.rows, seed=args.seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(
            format_error(
                exc,
                "Re-run the command and verify your Python environment dependencies.",
            )
        ) from exc

    print(f"Saved frontier demo data to: {out_path}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(frame.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
