"""Common helpers for example scripts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = EXAMPLES_DIR / "data" / "california_housing.csv"
DEFAULT_BINARY_DATA_PATH = EXAMPLES_DIR / "data" / "breast_cancer_binary.csv"
DEFAULT_OUT_DIR = EXAMPLES_DIR / "out"
DEFAULT_TARGET = "target"


def make_timestamp_dir(root: str | Path) -> Path:
    """Create a timestamped output directory and return it."""
    base = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def to_jsonable(value: Any) -> Any:
    """Convert objects used in examples into JSON-serializable values."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def save_json(path: str | Path, payload: Any) -> None:
    """Write JSON payload with deterministic formatting."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Write YAML payload preserving key order."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def format_error(error: Exception, hint: str) -> str:
    """Format a user-facing error with a concrete next action."""
    return f"{error}\nHint: {hint}"
