from __future__ import annotations

from veldra.gui.template_service import count_yaml_changes


def test_count_yaml_changes_identical() -> None:
    yaml_text = "config_version: 1\na: 1\n"
    assert count_yaml_changes(yaml_text, yaml_text) == 0


def test_count_yaml_changes_detects_delta() -> None:
    left = "config_version: 1\na: 1\nb:\n  c: 2\n"
    right = "config_version: 1\na: 2\nb:\n  c: 2\n"
    assert count_yaml_changes(left, right) >= 1


def test_count_yaml_changes_fallback_for_invalid_yaml() -> None:
    left = "{bad"
    right = "{worse"
    assert count_yaml_changes(left, right) >= 1
