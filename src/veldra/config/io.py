"""RunConfig serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from veldra.config.models import RunConfig


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RunConfig YAML must deserialize to a mapping object.")
    return RunConfig.model_validate(raw)


def save_run_config(config: RunConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = config.model_dump(mode="json", exclude_none=True)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
