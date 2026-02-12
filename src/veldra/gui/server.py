"""GUI server entrypoint."""

from __future__ import annotations

import argparse
import logging
import os

from veldra.api.logging import log_event

LOGGER = logging.getLogger("veldra.gui")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Veldra Dash GUI.")
    parser.add_argument("--host", default=os.getenv("VELDRA_GUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VELDRA_GUI_PORT", "8050")))
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("VELDRA_GUI_DEBUG", "0") == "1",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        from veldra.gui.app import create_app
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GUI dependencies are not installed. Run: uv sync --extra gui"
        ) from exc

    app = create_app()
    log_event(
        LOGGER,
        logging.INFO,
        "gui startup",
        run_id="gui",
        artifact_path=None,
        task_type="gui",
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
