from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_readme_has_required_sections() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## Why VeldraML?" in readme
    assert "## From Quick Start to Production" in readme
    assert "## RunConfig Reference (Complete)" in readme
    assert "<!-- RUNCONFIG_REF:START -->" in readme
    assert "<!-- RUNCONFIG_REF:END -->" in readme


def test_runconfig_reference_block_is_up_to_date() -> None:
    cmd = [sys.executable, "scripts/generate_runconfig_reference.py", "--check"]
    result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
