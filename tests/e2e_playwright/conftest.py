from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def phase26_manifest() -> dict:
    path = Path("notebooks/phase26_2_execution_manifest.json")
    if not path.exists():
        pytest.skip("phase26_2_execution_manifest.json is missing")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def phase26_entries(phase26_manifest: dict) -> dict[str, dict]:
    entries = phase26_manifest.get("entries") or []
    return {str(entry.get("uc")): entry for entry in entries}


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
def sample_data_paths(phase26_entries: dict[str, dict]) -> dict[str, str]:
    uc1_outputs = phase26_entries.get("UC-1", {}).get("outputs", [])
    uc8_outputs = phase26_entries.get("UC-8", {}).get("outputs", [])
    return {
        "uc1_train": str(uc1_outputs[0]) if len(uc1_outputs) > 0 else "",
        "uc1_test": str(uc1_outputs[1]) if len(uc1_outputs) > 1 else "",
        "uc8_ok": str(uc8_outputs[0]) if len(uc8_outputs) > 0 else "",
        "uc8_bad": str(uc8_outputs[1]) if len(uc8_outputs) > 1 else "",
    }
