from __future__ import annotations

from veldra.config.migrate import migrate_run_config_payload


def _minimal_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "train.csv", "target": "y"},
    }


def test_migrate_payload_normalizes_and_marks_changed() -> None:
    normalized, result = migrate_run_config_payload(_minimal_payload())
    assert normalized["config_version"] == 1
    assert result.source_version == 1
    assert result.target_version == 1
    assert result.changed is True
    assert result.input_path is None
    assert result.output_path is None


def test_migrate_payload_changed_false_after_second_pass() -> None:
    normalized_once, _ = migrate_run_config_payload(_minimal_payload())
    normalized_twice, result = migrate_run_config_payload(normalized_once)
    assert normalized_twice == normalized_once
    assert result.changed is False
