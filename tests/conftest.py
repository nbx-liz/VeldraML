"""Pytest shared setup."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-local tmp_path to avoid permission issues in this environment."""
    temp_root = REPO_ROOT / "tmp_tests"
    temp_root.mkdir(parents=True, exist_ok=True)
    created = temp_root / f"case_{uuid4().hex}"
    created.mkdir(parents=True, exist_ok=False)
    try:
        yield created
    finally:
        shutil.rmtree(created, ignore_errors=True)
