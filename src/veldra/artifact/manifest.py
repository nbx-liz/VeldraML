"""Artifact manifest model and builders."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from veldra.config.models import RunConfig

TRACKED_DEPENDENCIES = [
    "lightgbm",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "pyyaml",
    "scikit-learn",
]


class Manifest(BaseModel):
    """Metadata to preserve reproducibility for each artifact."""

    model_config = ConfigDict(extra="forbid")
    manifest_version: int = 1
    project_version: str
    run_id: str
    task_type: str
    config_version: int
    config_hash: str
    python_version: str
    dependencies: dict[str, str] = Field(default_factory=dict)
    created_at_utc: str


def _config_hash(run_config: RunConfig) -> str:
    payload: dict[str, Any] = run_config.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_manifest(run_config: RunConfig, run_id: str, project_version: str) -> Manifest:
    return Manifest(
        project_version=project_version,
        run_id=run_id,
        task_type=run_config.task.type,
        config_version=run_config.config_version,
        config_hash=_config_hash(run_config),
        python_version=platform.python_version(),
        dependencies=_dependency_versions(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
