from __future__ import annotations

import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.gui.services import migrate_config_file_via_gui, migrate_config_from_yaml

_SOURCE = """
config_version: 1
task:
  type: regression
data:
  path: examples/data/california_housing.csv
  target: MedHouseVal
split:
  type: kfold
  n_splits: 5
  seed: 42
export:
  artifact_dir: artifacts
""".strip()


def test_migrate_config_from_yaml_preview() -> None:
    normalized_yaml, diff_text = migrate_config_from_yaml(_SOURCE, target_version=1)
    assert "config_version: 1" in normalized_yaml
    assert isinstance(diff_text, str)


def test_migrate_config_file_contract(tmp_path) -> None:
    src = tmp_path / "run.yaml"
    # out = tmp_path / "run.migrated.yaml" # output path not supported in GUI wrapper currently
    src.write_text(_SOURCE, encoding="utf-8")

    # Wrapper returns a string message
    result_msg = migrate_config_file_via_gui(
        input_path=str(src),
        # output_path=str(out),
        target_version=1,
    )
    assert "Migration successful" in result_msg
    # assert out.exists() 
    # Logic note: migrate_run_config_file by default refuses to overwrite. 
    # But wrapper catches exception. If src exists, output defaults to src.migrated.yaml?
    # services.py wrapper calls migrate_run_config_file(input_path, target_version...).
    # Default output path is input_path.stem + .migrated.yaml.
    # So we should check that file.
    
    expected_out = src.with_name("run.migrated.yaml")
    assert expected_out.exists()

