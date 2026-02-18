from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from veldra.config.io import load_run_config, save_run_config
from veldra.config.models import RunConfig


def _run_config(tmp_path: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(tmp_path / "train.csv"), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 9},
            "postprocess": {"calibration": "platt"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )


def test_save_then_load_run_config_roundtrip(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    path = tmp_path / "configs" / "run_config.yaml"
    save_run_config(config, path)
    loaded = load_run_config(path)
    assert loaded.model_dump(mode="json") == config.model_dump(mode="json")


def test_save_run_config_creates_parent_directories(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    path = tmp_path / "deep" / "nested" / "run.yaml"
    save_run_config(config, path)
    assert path.exists()
    assert path.parent.exists()


def test_load_run_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping object"):
        load_run_config(path)


def test_load_run_config_raises_for_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("config_version: [1\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_run_config(path)


def test_load_run_config_raises_for_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run_config(tmp_path / "missing.yaml")
