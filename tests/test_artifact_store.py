from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from veldra.api.exceptions import VeldraArtifactError
from veldra.artifact.manifest import build_manifest
from veldra.artifact.store import load_artifact, save_artifact
from veldra.config.io import save_run_config
from veldra.config.models import RunConfig


def _run_config(tmp_path: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(tmp_path / "train.csv"), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )


def test_save_load_roundtrip_persists_optional_payloads(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    manifest = build_manifest(config, run_id="rid_store", project_version="0.1.0")
    artifact_dir = tmp_path / "artifact_store_roundtrip"

    cv_results = pd.DataFrame({"fold": [1, 2], "rmse": [0.2, 0.3]})
    calibration_curve = pd.DataFrame({"prob_pred": [0.1, 0.8], "prob_true": [0.2, 0.7]})
    threshold_curve = pd.DataFrame({"threshold": [0.2, 0.5], "f1": [0.6, 0.7]})
    observation_table = pd.DataFrame({"fold_id": [1, 2], "prediction": [1.2, 1.8]})

    save_artifact(
        path=artifact_dir,
        run_config=config,
        manifest=manifest,
        feature_schema={"feature_names": ["x1", "x2"], "target": "target"},
        model_text="model-text",
        metrics={"mean": {"rmse": 0.25}},
        cv_results=cv_results,
        calibrator={"kind": "dummy"},
        calibration_curve=calibration_curve,
        threshold={"policy": "fixed", "value": 0.5},
        threshold_curve=threshold_curve,
        training_history={"final_model": {"best_iteration": 3}},
        observation_table=observation_table,
    )

    loaded_config, loaded_manifest, feature_schema, extras = load_artifact(artifact_dir)
    assert loaded_config.model_dump(mode="json") == config.model_dump(mode="json")
    assert loaded_manifest.run_id == "rid_store"
    assert feature_schema == {"feature_names": ["x1", "x2"], "target": "target"}
    assert extras["model_text"] == "model-text"
    assert extras["metrics"] == {"mean": {"rmse": 0.25}}
    assert extras["cv_results"].equals(cv_results)
    assert extras["calibrator"] == {"kind": "dummy"}
    assert extras["calibration_curve"].equals(calibration_curve)
    assert extras["threshold"] == {"policy": "fixed", "value": 0.5}
    assert extras["threshold_curve"].equals(threshold_curve)
    assert extras["training_history"] == {"final_model": {"best_iteration": 3}}
    assert extras["observation_table"].equals(observation_table)


def test_load_artifact_returns_none_for_missing_optional_payloads(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    manifest = build_manifest(config, run_id="rid_min", project_version="0.1.0")
    artifact_dir = tmp_path / "artifact_store_minimal"
    save_artifact(
        path=artifact_dir,
        run_config=config,
        manifest=manifest,
        feature_schema={"feature_names": ["x1"], "target": "target"},
    )

    _, _, _, extras = load_artifact(artifact_dir)
    assert extras == {
        "model_text": None,
        "metrics": None,
        "cv_results": None,
        "calibrator": None,
        "calibration_curve": None,
        "threshold": None,
        "threshold_curve": None,
        "training_history": None,
        "observation_table": None,
    }


def test_load_artifact_rejects_missing_required_files(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact_missing"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(VeldraArtifactError, match="missing required file"):
        load_artifact(artifact_dir)


def test_load_artifact_rejects_non_object_feature_schema(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    manifest = build_manifest(config, run_id="rid_bad_schema", project_version="0.1.0")
    artifact_dir = tmp_path / "artifact_bad_schema"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    save_run_config(config, artifact_dir / "run_config.yaml")
    (artifact_dir / "feature_schema.json").write_text(json.dumps(["x1", "x2"]), encoding="utf-8")

    with pytest.raises(
        VeldraArtifactError,
        match="feature_schema.json must deserialize to an object",
    ):
        load_artifact(artifact_dir)


def test_load_artifact_wraps_manifest_parse_failure(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    artifact_dir = tmp_path / "artifact_bad_manifest"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    save_run_config(config, artifact_dir / "run_config.yaml")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps({"feature_names": ["x1"]}),
        encoding="utf-8",
    )

    with pytest.raises(VeldraArtifactError, match="Failed to load artifact"):
        load_artifact(artifact_dir)
