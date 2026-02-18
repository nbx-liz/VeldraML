from __future__ import annotations

import pytest

import veldra.config as config_pkg


def test_config_package_getattr_resolves_known_exports() -> None:
    assert config_pkg.load_run_config.__name__ == "load_run_config"
    assert config_pkg.save_run_config.__name__ == "save_run_config"
    assert config_pkg.migrate_run_config_payload.__name__ == "migrate_run_config_payload"
    assert config_pkg.migrate_run_config_file.__name__ == "migrate_run_config_file"
    assert config_pkg.MigrationResult.__name__ == "MigrationResult"
    assert config_pkg.RunConfig.__name__ == "RunConfig"
    assert config_pkg.TaskConfig.__name__ == "TaskConfig"


def test_config_package_getattr_unknown_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = config_pkg.not_existing_export_name
