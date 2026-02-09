"""Artifact persistence functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from veldra.api.exceptions import VeldraArtifactError
from veldra.artifact.manifest import Manifest
from veldra.config.io import load_run_config, save_run_config
from veldra.config.models import RunConfig


def save_artifact(
    path: str | Path,
    run_config: RunConfig,
    manifest: Manifest,
    feature_schema: dict[str, Any],
) -> None:
    artifact_dir = Path(path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    save_run_config(run_config, artifact_dir / "run_config.yaml")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_artifact(path: str | Path) -> tuple[RunConfig, Manifest, dict[str, Any]]:
    artifact_dir = Path(path)
    manifest_path = artifact_dir / "manifest.json"
    run_config_path = artifact_dir / "run_config.yaml"
    schema_path = artifact_dir / "feature_schema.json"

    missing = [p.name for p in [manifest_path, run_config_path, schema_path] if not p.exists()]
    if missing:
        raise VeldraArtifactError(f"Artifact is missing required file(s): {', '.join(missing)}")

    try:
        manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        run_config = load_run_config(run_config_path)
        feature_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - safety net
        raise VeldraArtifactError(f"Failed to load artifact from {artifact_dir}") from exc

    if not isinstance(feature_schema, dict):
        raise VeldraArtifactError("feature_schema.json must deserialize to an object.")

    return run_config, manifest, feature_schema
