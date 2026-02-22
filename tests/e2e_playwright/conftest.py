from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest


def _require_path(path: Path, *, reason: str) -> str:
    if not path.exists():
        pytest.skip(f"{reason}: missing {path}")
    return str(path.resolve())


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(host: str, port: int, timeout_sec: float = 25.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


@pytest.fixture(scope="session")
def gui_base_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    work = tmp_path_factory.mktemp("gui_e2e")
    db_path = work / "jobs.sqlite3"
    try:
        port = _find_free_port()
    except OSError as exc:
        pytest.skip(f"Socket bind is unavailable in this environment: {exc}")
    host = "127.0.0.1"

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", ".uv_cache")
    env.setdefault("VELDRA_GUI_POLL_MS", "300")

    cmd = [
        "uv",
        "run",
        "veldra-gui",
        "--host",
        host,
        "--port",
        str(port),
        "--job-db-path",
        str(db_path),
        "--worker-poll-sec",
        "0.2",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    if not _wait_port(host, port):
        proc.terminate()
        try:
            _out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _out, err = proc.communicate()
        pytest.skip(f"Failed to boot GUI server for Playwright E2E: {err.strip()}")

    try:
        yield f"http://{host}:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def artifact_path_uc1() -> str:
    summary = Path("examples/out/phase35_uc01_regression_fit_evaluate/summary.json")
    _require_path(summary, reason="UC-1 summary")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    artifact_path = Path(str(payload.get("artifact_path", "")).strip())
    return _require_path(artifact_path, reason="UC-1 artifact")


@pytest.fixture(scope="session")
def artifact_root_uc1(artifact_path_uc1: str) -> str:
    return str(Path(artifact_path_uc1).parent)


@pytest.fixture(scope="session")
def sample_data_paths() -> dict[str, str]:
    return {
        "uc1_train": _require_path(
            Path("examples/out/phase35_uc01_regression_fit_evaluate/train.parquet"),
            reason="UC-1 train data",
        ),
        "uc1_test": _require_path(
            Path("examples/out/phase35_uc01_regression_fit_evaluate/test.parquet"),
            reason="UC-1 test data",
        ),
        "uc8_ok": _require_path(
            Path("examples/out/phase35_uc08_artifact_evaluate/latest.csv"),
            reason="UC-8 re-eval ok data",
        ),
        "uc8_bad": _require_path(
            Path("examples/out/phase35_uc08_artifact_evaluate/eval_table.csv"),
            reason="UC-8 re-eval bad data",
        ),
    }
