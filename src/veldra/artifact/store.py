"""Artifact persistence functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from veldra.api.exceptions import VeldraArtifactError
from veldra.artifact.manifest import Manifest
from veldra.config.io import load_run_config, save_run_config
from veldra.config.models import RunConfig


def save_artifact(
    path: str | Path,
    run_config: RunConfig,
    manifest: Manifest,
    feature_schema: dict[str, Any],
    model_text: str | None = None,
    metrics: dict[str, Any] | None = None,
    cv_results: pd.DataFrame | None = None,
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
    if model_text is not None:
        (artifact_dir / "model.lgb.txt").write_text(model_text, encoding="utf-8")
    if metrics is not None:
        (artifact_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if cv_results is not None:
        cv_results.to_parquet(artifact_dir / "cv_results.parquet", index=False)


def load_artifact(path: str | Path) -> tuple[RunConfig, Manifest, dict[str, Any], dict[str, Any]]:
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

    extras: dict[str, Any] = {
        "model_text": None,
        "metrics": None,
        "cv_results": None,
    }
    model_path = artifact_dir / "model.lgb.txt"
    metrics_path = artifact_dir / "metrics.json"
    cv_results_path = artifact_dir / "cv_results.parquet"
    if model_path.exists():
        extras["model_text"] = model_path.read_text(encoding="utf-8")
    if metrics_path.exists():
        extras["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if cv_results_path.exists():
        extras["cv_results"] = pd.read_parquet(cv_results_path)

    return run_config, manifest, feature_schema, extras
