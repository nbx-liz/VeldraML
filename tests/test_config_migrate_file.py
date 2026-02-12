from __future__ import annotations

from pathlib import Path

import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.migrate import migrate_run_config_file


def _write_config(path: Path) -> None:
    path.write_text(
        "config_version: 1\n"
        "task:\n"
        "  type: regression\n"
        "data:\n"
        "  path: train.csv\n"
        "  target: y\n",
        encoding="utf-8",
    )


def test_migrate_file_writes_default_output_path(tmp_path) -> None:
    src = tmp_path / "run.yaml"
    _write_config(src)
    result = migrate_run_config_file(src)
    assert result.output_path == str(tmp_path / "run.migrated.yaml")
    assert (tmp_path / "run.migrated.yaml").exists()


def test_migrate_file_rejects_existing_output(tmp_path) -> None:
    src = tmp_path / "run.yaml"
    out = tmp_path / "custom.yaml"
    _write_config(src)
    out.write_text("already exists\n", encoding="utf-8")
    with pytest.raises(VeldraValidationError, match="Refusing to overwrite existing file"):
        migrate_run_config_file(src, output_path=out)


def test_migrate_file_rejects_malformed_yaml_and_non_mapping(tmp_path) -> None:
    malformed = tmp_path / "bad.yaml"
    malformed.write_text("key: [\n", encoding="utf-8")
    with pytest.raises(VeldraValidationError, match="Config YAML parse error"):
        migrate_run_config_file(malformed)

    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(VeldraValidationError, match="must deserialize to a mapping object"):
        migrate_run_config_file(non_mapping)


def test_migrate_file_rejects_missing_input(tmp_path) -> None:
    with pytest.raises(VeldraValidationError, match="does not exist"):
        migrate_run_config_file(tmp_path / "missing.yaml")
