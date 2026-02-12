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
    normalized_yaml, diff_text, meta = migrate_config_from_yaml(_SOURCE, target_version=1)
    assert "config_version: 1" in normalized_yaml
    assert isinstance(diff_text, str)
    assert meta["target_version"] == 1
    assert "changed" in meta


def test_migrate_config_file_contract(tmp_path) -> None:
    src = tmp_path / "run.yaml"
    out = tmp_path / "run.migrated.yaml"
    src.write_text(_SOURCE, encoding="utf-8")

    result = migrate_config_file_via_gui(
        input_path=str(src),
        output_path=str(out),
        target_version=1,
    )
    assert result["output_path"] == str(out)
    assert out.exists()

    with pytest.raises(VeldraValidationError):
        migrate_config_file_via_gui(
            input_path=str(src),
            output_path=str(out),
            target_version=1,
        )
