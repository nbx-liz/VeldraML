"""Run export demo for an existing artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - import path depends on launch style
    from examples.common import DEFAULT_OUT_DIR, format_error, make_timestamp_dir, save_json
except ModuleNotFoundError:  # pragma: no cover
    from common import DEFAULT_OUT_DIR, format_error, make_timestamp_dir, save_json

from veldra.api import Artifact, export


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True, help="Path to saved artifact directory.")
    parser.add_argument(
        "--format",
        default="python",
        choices=["python", "onnx"],
        help="Export format.",
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
        export_result = export(artifact, format=args.format)
        run_dir = make_timestamp_dir(args.out_dir)
        save_json(run_dir / "export_result.json", export_result)
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(
            format_error(
                exc,
                "Verify artifact path and optional ONNX dependencies, then retry.",
            )
        ) from exc

    print(f"format: {export_result.format}")
    print(f"export_path: {export_result.path}")
    print(f"validation_passed: {export_result.metadata.get('validation_passed')}")
    print(f"validation_report: {export_result.metadata.get('validation_report')}")
    print(f"output_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
