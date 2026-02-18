from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HEAVY_MODULES = [
    "veldra.api.runner",
    "veldra.api.artifact",
    "veldra.data",
    "lightgbm",
    "optuna",
    "sklearn",
]


def _probe_import_state(module_name: str) -> dict[str, bool]:
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import json,sys;"
        f"sys.path.insert(0,{repo_root.as_posix()!r});"
        f"import {module_name};"
        f"mods={_HEAVY_MODULES!r};"
        "print(json.dumps({m:(m in sys.modules) for m in mods}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def test_cold_import_gui_app_does_not_load_heavy_modules() -> None:
    loaded = _probe_import_state("veldra.gui.app")
    assert all(not is_loaded for is_loaded in loaded.values()), loaded


def test_cold_import_gui_services_does_not_load_heavy_modules() -> None:
    loaded = _probe_import_state("veldra.gui.services")
    assert all(not is_loaded for is_loaded in loaded.values()), loaded
