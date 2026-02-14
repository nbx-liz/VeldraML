"""GUI server entrypoint."""

from __future__ import annotations

import argparse
import atexit
import logging
import os

from veldra.api.logging import log_event
from veldra.gui.job_store import GuiJobStore
from veldra.gui.services import set_gui_runtime, stop_gui_runtime
from veldra.gui.worker import GuiWorker

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
    parser.add_argument(
        "--job-db-path",
        default=os.getenv("VELDRA_GUI_JOB_DB_PATH", ".veldra_gui/jobs.sqlite3"),
    )
    parser.add_argument(
        "--worker-poll-sec",
        type=float,
        default=float(os.getenv("VELDRA_GUI_WORKER_POLL_SEC", "0.5")),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        from veldra.gui.app import create_app
    except ModuleNotFoundError as exc:
        raise RuntimeError("GUI dependencies are not installed. Run: uv sync --extra gui") from exc

    store = GuiJobStore(args.job_db_path)
    worker = GuiWorker(store, poll_interval_sec=args.worker_poll_sec)
    set_gui_runtime(job_store=store, worker=worker)
    worker.start()
    atexit.register(stop_gui_runtime)

    app = create_app()
    log_event(
        LOGGER,
        logging.INFO,
        "gui startup",
        run_id="gui",
        artifact_path=None,
        task_type="gui",
        event="gui startup",
        host=args.host,
        port=args.port,
        debug=args.debug,
        job_db_path=str(args.job_db_path),
    )
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    finally:
        stop_gui_runtime()


if __name__ == "__main__":
    main()
