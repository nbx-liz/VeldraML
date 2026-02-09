"""Evaluate an existing frontier artifact using labeled DataFrame input."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - import path depends on how the script is launched
    from examples.common import (
        DEFAULT_FRONTIER_DATA_PATH,
        DEFAULT_OUT_DIR,
        format_error,
        make_timestamp_dir,
        save_json,
    )
except ModuleNotFoundError:  # pragma: no cover
    from common import (
        DEFAULT_FRONTIER_DATA_PATH,
        DEFAULT_OUT_DIR,
        format_error,
        make_timestamp_dir,
        save_json,
    )

from veldra.api import Artifact, evaluate
from veldra.data import load_tabular_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Path to saved artifact directory.")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_FRONTIER_DATA_PATH),
        help="Evaluation data path (CSV/Parquet) with target column.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output root directory (default: examples/out).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        artifact = Artifact.load(Path(args.artifact_path))
        frame = load_tabular_data(args.data_path)
        eval_result = evaluate(artifact, frame)

        run_dir = make_timestamp_dir(args.out_dir)
        save_json(run_dir / "eval_only_result.json", eval_result)
    except Exception as exc:  # pragma: no cover - exercised via CLI failure only
        raise SystemExit(
            format_error(
                exc,
                "Verify artifact path and labeled frontier data, then retry.",
            )
        ) from exc

    print(f"pinball: {eval_result.metrics['pinball']:.6f}")
    print(f"mae: {eval_result.metrics['mae']:.6f}")
    print(f"mean_u_hat: {eval_result.metrics['mean_u_hat']:.6f}")
    print(f"coverage: {eval_result.metrics['coverage']:.6f}")
    print(f"n_rows: {eval_result.metadata['n_rows']}")
    print(f"output_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
