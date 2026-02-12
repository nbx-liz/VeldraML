from __future__ import annotations

import pytest

from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError
from veldra.config.migrate import migrate_run_config_payload


def _payload(version: int) -> dict:
    return {
        "config_version": version,
        "task": {"type": "regression"},
        "data": {"path": "train.csv", "target": "y"},
    }


def test_migrate_payload_rejects_non_v1_source() -> None:
    with pytest.raises(VeldraNotImplementedError, match="config_version=2"):
        migrate_run_config_payload(_payload(2))


def test_migrate_payload_rejects_non_v1_target_version() -> None:
    with pytest.raises(VeldraNotImplementedError, match="target_version=2"):
        migrate_run_config_payload(_payload(1), target_version=2)


def test_migrate_payload_rejects_non_int_config_version() -> None:
    payload = _payload(1)
    payload["config_version"] = "1"
    with pytest.raises(VeldraValidationError, match="config_version must be an integer"):
        migrate_run_config_payload(payload)
