"""Artifact persistence functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
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
    calibrator: Any | None = None,
    calibration_curve: pd.DataFrame | None = None,
    threshold: dict[str, Any] | None = None,
    threshold_curve: pd.DataFrame | None = None,
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
    if calibrator is not None:
        joblib.dump(calibrator, artifact_dir / "calibrator.pkl")
    if calibration_curve is not None:
        calibration_curve.to_csv(artifact_dir / "calibration_curve.csv", index=False)
    if threshold is not None:
        (artifact_dir / "threshold.json").write_text(
            json.dumps(threshold, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if threshold_curve is not None:
        threshold_curve.to_csv(artifact_dir / "threshold_curve.csv", index=False)


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
        "calibrator": None,
        "calibration_curve": None,
        "threshold": None,
        "threshold_curve": None,
    }
    model_path = artifact_dir / "model.lgb.txt"
    metrics_path = artifact_dir / "metrics.json"
    cv_results_path = artifact_dir / "cv_results.parquet"
    calibrator_path = artifact_dir / "calibrator.pkl"
    calibration_curve_path = artifact_dir / "calibration_curve.csv"
    threshold_path = artifact_dir / "threshold.json"
    threshold_curve_path = artifact_dir / "threshold_curve.csv"
    if model_path.exists():
        extras["model_text"] = model_path.read_text(encoding="utf-8")
    if metrics_path.exists():
        extras["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if cv_results_path.exists():
        extras["cv_results"] = pd.read_parquet(cv_results_path)
    if calibrator_path.exists():
        extras["calibrator"] = joblib.load(calibrator_path)
    if calibration_curve_path.exists():
        extras["calibration_curve"] = pd.read_csv(calibration_curve_path)
    if threshold_path.exists():
        extras["threshold"] = json.loads(threshold_path.read_text(encoding="utf-8"))
    if threshold_curve_path.exists():
        extras["threshold_curve"] = pd.read_csv(threshold_curve_path)

    return run_config, manifest, feature_schema, extras
