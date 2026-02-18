from __future__ import annotations

from veldra.gui import template_service

REQUIRED_TEMPLATE_IDS = {
    "regression_baseline",
    "binary_balanced",
    "multiclass_standard",
    "causal_dr_panel",
    "tuning_standard",
}


def test_builtin_templates_have_required_ids_and_unique() -> None:
    templates = template_service.list_builtin_templates()
    ids = [item["template_id"] for item in templates]
    assert REQUIRED_TEMPLATE_IDS.issubset(set(ids))
    assert len(ids) == len(set(ids))


def test_builtin_templates_are_all_valid_runconfigs() -> None:
    invalid = template_service.validate_builtin_templates()
    assert invalid == []


def test_load_builtin_template_yaml() -> None:
    yaml_text = template_service.load_builtin_template_yaml("regression_baseline")
    assert "config_version: 1" in yaml_text
    template_service.validate_template_yaml(yaml_text)
