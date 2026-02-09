"""Prepare Breast Cancer CSV used by binary demo scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

try:  # pragma: no cover - import path depends on launch style
    from examples.common import DEFAULT_BINARY_DATA_PATH, DEFAULT_TARGET, format_error
except ModuleNotFoundError:  # pragma: no cover
    from common import DEFAULT_BINARY_DATA_PATH, DEFAULT_TARGET, format_error


def build_breast_cancer_frame() -> pd.DataFrame:
    """Load sklearn breast cancer dataset and normalize target name."""
    fetched = load_breast_cancer(as_frame=True)
    frame = fetched.frame.copy()
    if "target" in frame.columns and DEFAULT_TARGET != "target":
        frame = frame.rename(columns={"target": DEFAULT_TARGET})
    elif DEFAULT_TARGET not in frame.columns and hasattr(fetched, "target"):
        frame[DEFAULT_TARGET] = fetched.target
    if DEFAULT_TARGET not in frame.columns:
        raise ValueError("Could not determine target column from breast cancer dataset.")
    return frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_BINARY_DATA_PATH),
        help="Output CSV path (default: examples/data/breast_cancer_binary.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out_path)

    try:
        frame = build_breast_cancer_frame()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(
            format_error(
                exc,
                "Re-run the command and verify your Python environment dependencies.",
            )
        ) from exc

    print(f"Saved binary demo data to: {out_path}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(frame.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
