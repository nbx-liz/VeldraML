from __future__ import annotations

import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.gui.services import load_config_yaml, save_config_yaml, validate_config


def test_validate_config_success() -> None:
    config = validate_config(
        """
config_version: 1
task:
  type: regression
data:
  path: examples/data/california_housing.csv
  target: MedHouseVal
        """.strip()
    )
    assert config.task.type == "regression"
    assert config.data.target == "MedHouseVal"


def test_validate_config_invalid_payload_type() -> None:
    with pytest.raises(VeldraValidationError):
        validate_config("- a\n- b")


def test_save_and_load_config_yaml_roundtrip(tmp_path) -> None:
    path = tmp_path / "cfg" / "run.yaml"
    source = "config_version: 1\ntask:\n  type: regression\ndata:\n  path: train.csv\n  target: y\n"
    written = save_config_yaml(str(path), source)
    assert written == str(path)
    loaded = load_config_yaml(str(path))
    assert loaded == source
