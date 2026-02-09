"""Prepare Iris CSV used by multiclass demo scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

try:  # pragma: no cover - import path depends on launch style
    from examples.common import DEFAULT_MULTICLASS_DATA_PATH, DEFAULT_TARGET, format_error
except ModuleNotFoundError:  # pragma: no cover
    from common import DEFAULT_MULTICLASS_DATA_PATH, DEFAULT_TARGET, format_error


def build_iris_frame() -> pd.DataFrame:
    """Load sklearn iris dataset and normalize target values to class names."""
    fetched = load_iris(as_frame=True)
    frame = fetched.frame.copy()
    if DEFAULT_TARGET not in frame.columns:
        if "target" in frame.columns:
            frame = frame.rename(columns={"target": DEFAULT_TARGET})
        else:
            raise ValueError("Could not determine target column from iris dataset.")

    if hasattr(fetched, "target_names"):
        label_map = {idx: str(name) for idx, name in enumerate(fetched.target_names)}
        frame[DEFAULT_TARGET] = frame[DEFAULT_TARGET].map(label_map)
    if frame[DEFAULT_TARGET].isna().any():
        raise ValueError("Target mapping produced null values.")
    return frame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_MULTICLASS_DATA_PATH),
        help="Output CSV path (default: examples/data/iris_multiclass.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out_path)

    try:
        frame = build_iris_frame()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(
            format_error(
                exc,
                "Re-run the command and verify your Python environment dependencies.",
            )
        ) from exc

    print(f"Saved multiclass demo data to: {out_path}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(frame.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
