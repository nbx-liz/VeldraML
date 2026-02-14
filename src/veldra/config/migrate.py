"""RunConfig migration helpers (MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError
from veldra.config.models import RunConfig


@dataclass(slots=True)
class MigrationResult:
    input_path: str | None
    output_path: str | None
    source_version: int
    target_version: int
    changed: bool
    warnings: list[str] = field(default_factory=list)


def _ensure_target_version_supported(target_version: int) -> None:
    if target_version != 1:
        raise VeldraNotImplementedError(
            f"target_version={target_version} is not supported yet. "
            "This phase supports only target_version=1."
        )


def _normalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    source_version_raw = payload.get("config_version")
    if not isinstance(source_version_raw, int):
        raise VeldraValidationError("config_version must be an integer.")
    if source_version_raw != 1:
        raise VeldraNotImplementedError(
            f"config_version={source_version_raw} is not supported yet. "
            "This phase supports only config_version=1."
        )

    try:
        config = RunConfig.model_validate(payload)
    except ValidationError as exc:
        raise VeldraValidationError(f"Invalid RunConfig: {exc}") from exc

    normalized = config.model_dump(mode="json", exclude_none=True)
    return normalized, source_version_raw


def migrate_run_config_payload(
    payload: dict[str, Any],
    *,
    target_version: int = 1,
) -> tuple[dict[str, Any], MigrationResult]:
    """Validate and normalize a RunConfig payload."""
    _ensure_target_version_supported(target_version)
    if not isinstance(payload, dict):
        raise VeldraValidationError("RunConfig payload must be a mapping object.")

    normalized, source_version = _normalize_payload(payload)
    changed = normalized != payload
    result = MigrationResult(
        input_path=None,
        output_path=None,
        source_version=source_version,
        target_version=target_version,
        changed=changed,
        warnings=[],
    )
    return normalized, result


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.migrated.yaml")


def migrate_run_config_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    target_version: int = 1,
) -> MigrationResult:
    """Validate and normalize a RunConfig YAML file."""
    _ensure_target_version_supported(target_version)

    source_path = Path(input_path)
    if not source_path.exists():
        raise VeldraValidationError(f"Config file does not exist: {source_path}")

    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VeldraValidationError(f"Config YAML parse error: {exc}") from exc

    if not isinstance(raw, dict):
        raise VeldraValidationError("RunConfig YAML must deserialize to a mapping object.")

    normalized, source_version = _normalize_payload(raw)

    destination = (
        Path(output_path) if output_path is not None else _default_output_path(source_path)
    )
    if destination.exists():
        raise VeldraValidationError(
            f"Refusing to overwrite existing file: {destination}. Choose a different --output path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return MigrationResult(
        input_path=str(source_path),
        output_path=str(destination),
        source_version=source_version,
        target_version=target_version,
        changed=(normalized != raw),
        warnings=[],
    )
