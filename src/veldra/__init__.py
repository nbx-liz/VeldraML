"""Veldra package root."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from typing import Sequence

__version__ = "0.1.0"

LOGGER = logging.getLogger("veldra.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veldra", description="Veldra command line tools.")
    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="RunConfig utilities.")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    migrate_parser = config_subparsers.add_parser("migrate", help="Normalize config YAML.")
    migrate_parser.add_argument("--input", required=True, help="Input RunConfig YAML path.")
    migrate_parser.add_argument(
        "--output",
        required=False,
        help="Output path. Default: <input_stem>.migrated.yaml",
    )
    migrate_parser.add_argument(
        "--target-version",
        type=int,
        default=1,
        help="Migration target version (MVP supports only 1).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from veldra.api.exceptions import VeldraError
    from veldra.api.logging import log_event
    from veldra.config.migrate import migrate_run_config_file

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "config" or args.config_command != "migrate":
        parser.print_help()
        return

    try:
        result = migrate_run_config_file(
            input_path=args.input,
            output_path=args.output,
            target_version=args.target_version,
        )
    except VeldraError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    log_event(
        LOGGER,
        logging.INFO,
        "config migrate completed",
        run_id="config-migrate",
        artifact_path=None,
        task_type="config",
        input_path=result.input_path,
        output_path=result.output_path,
        source_version=result.source_version,
        target_version=result.target_version,
        changed=result.changed,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
