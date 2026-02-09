"""Prepare California Housing CSV used by demo scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

try:  # pragma: no cover - import path depends on how the script is launched
    from examples.common import DEFAULT_DATA_PATH, DEFAULT_TARGET, format_error
except ModuleNotFoundError:  # pragma: no cover
    from common import DEFAULT_DATA_PATH, DEFAULT_TARGET, format_error


def build_california_frame() -> pd.DataFrame:
    """Fetch California Housing and normalize target column name."""
    data_home = DEFAULT_DATA_PATH.parent / "sklearn_data"
    data_home.mkdir(parents=True, exist_ok=True)
    fetched = fetch_california_housing(as_frame=True, data_home=str(data_home))
    frame = fetched.frame.copy()
    if "MedHouseVal" in frame.columns:
        frame = frame.rename(columns={"MedHouseVal": DEFAULT_TARGET})
    elif DEFAULT_TARGET not in frame.columns and hasattr(fetched, "target"):
        frame[DEFAULT_TARGET] = fetched.target
    if DEFAULT_TARGET not in frame.columns:
        raise ValueError("Could not determine target column from fetched dataset.")
    return frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_DATA_PATH),
        help="Output CSV path (default: examples/data/california_housing.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out_path)

    try:
        frame = build_california_frame()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    except Exception as exc:  # pragma: no cover - exercised via CLI failure only
        raise SystemExit(
            format_error(
                exc,
                "Re-run the command and verify internet/network access for dataset download.",
            )
        ) from exc

    print(f"Saved demo data to: {out_path}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(frame.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
