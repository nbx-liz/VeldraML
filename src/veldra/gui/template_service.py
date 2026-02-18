"""Template and local-config helpers for GUI Phase30."""

from __future__ import annotations

import difflib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from veldra.config.models import RunConfig

_TEMPLATE_DIR = Path(__file__).with_suffix("").parent / "templates"
_MAX_CUSTOM_SLOTS = 10


@dataclass(frozen=True)
class TemplateMeta:
    template_id: str
    name: str
    description: str
    path: Path


_BUILTIN_TEMPLATES: tuple[TemplateMeta, ...] = (
    TemplateMeta(
        "regression_baseline",
        "Regression Baseline",
        "汎用回帰の初期設定",
        _TEMPLATE_DIR / "regression_baseline.yaml",
    ),
    TemplateMeta(
        "binary_balanced",
        "Binary Balanced",
        "不均衡二値分類向け初期設定",
        _TEMPLATE_DIR / "binary_balanced.yaml",
    ),
    TemplateMeta(
        "multiclass_standard",
        "Multiclass Standard",
        "多値分類の標準設定",
        _TEMPLATE_DIR / "multiclass_standard.yaml",
    ),
    TemplateMeta(
        "causal_dr_panel",
        "Causal DR Panel",
        "DR-DiD (panel) の最小設定",
        _TEMPLATE_DIR / "causal_dr_panel.yaml",
    ),
    TemplateMeta(
        "tuning_standard",
        "Tuning Standard",
        "標準ハイパーパラメータ探索設定",
        _TEMPLATE_DIR / "tuning_standard.yaml",
    ),
)


def list_builtin_templates() -> list[dict[str, str]]:
    return [
        {
            "template_id": item.template_id,
            "name": item.name,
            "description": item.description,
        }
        for item in _BUILTIN_TEMPLATES
    ]


def template_options() -> list[dict[str, str]]:
    return [
        {"label": f"{item.name} ({item.template_id})", "value": item.template_id}
        for item in _BUILTIN_TEMPLATES
    ]


def load_builtin_template_yaml(template_id: str) -> str:
    meta = next((item for item in _BUILTIN_TEMPLATES if item.template_id == template_id), None)
    if meta is None:
        raise ValueError(f"Unknown template_id: {template_id}")
    return meta.path.read_text(encoding="utf-8")


def _parse_yaml_mapping(yaml_text: str) -> dict[str, Any]:
    payload = yaml.safe_load(yaml_text)
    if not isinstance(payload, dict):
        raise ValueError("YAML must deserialize to a mapping.")
    return payload


def validate_template_yaml(yaml_text: str) -> None:
    payload = _parse_yaml_mapping(yaml_text)
    RunConfig.model_validate(payload)


def validate_builtin_templates() -> list[str]:
    invalid_ids: list[str] = []
    for item in _BUILTIN_TEMPLATES:
        try:
            validate_template_yaml(item.path.read_text(encoding="utf-8"))
        except Exception:
            invalid_ids.append(item.template_id)
    return invalid_ids


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_custom_slots(
    raw_slots: Any, *, max_slots: int = _MAX_CUSTOM_SLOTS
) -> list[dict[str, Any]]:
    if not isinstance(raw_slots, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_slots:
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("slot_id") or "").strip()
        name = str(item.get("name") or "").strip()
        yaml_text = str(item.get("yaml_text") or "")
        if not slot_id or not name or not yaml_text:
            continue
        normalized.append(
            {
                "slot_id": slot_id,
                "name": name,
                "yaml_text": yaml_text,
                "template_origin": str(item.get("template_origin") or "custom"),
                "created_at": str(item.get("created_at") or _now_iso()),
                "updated_at": str(item.get("updated_at") or _now_iso()),
            }
        )

    normalized.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    return normalized[:max_slots]


def _unique_copy_name(base_name: str, existing_names: set[str]) -> str:
    candidate = f"{base_name} (copy)"
    if candidate not in existing_names:
        return candidate
    idx = 2
    while True:
        cand = f"{base_name} (copy {idx})"
        if cand not in existing_names:
            return cand
        idx += 1


def save_custom_slot(
    raw_slots: Any,
    *,
    name: str,
    yaml_text: str,
    template_origin: str,
) -> list[dict[str, Any]]:
    slots = normalize_custom_slots(raw_slots)
    now = _now_iso()
    slots.insert(
        0,
        {
            "slot_id": uuid.uuid4().hex,
            "name": name.strip() or "Custom Config",
            "yaml_text": yaml_text,
            "template_origin": template_origin or "custom",
            "created_at": now,
            "updated_at": now,
        },
    )
    return normalize_custom_slots(slots)


def clone_custom_slot(raw_slots: Any, *, slot_id: str) -> list[dict[str, Any]]:
    slots = normalize_custom_slots(raw_slots)
    src = next((row for row in slots if str(row.get("slot_id")) == slot_id), None)
    if src is None:
        return slots
    names = {str(row.get("name") or "") for row in slots}
    clone_name = _unique_copy_name(str(src.get("name") or "Config"), names)
    return save_custom_slot(
        slots,
        name=clone_name,
        yaml_text=str(src.get("yaml_text") or ""),
        template_origin=str(src.get("template_origin") or "custom"),
    )


def load_custom_slot_yaml(raw_slots: Any, *, slot_id: str) -> str:
    slots = normalize_custom_slots(raw_slots)
    src = next((row for row in slots if str(row.get("slot_id")) == slot_id), None)
    if src is None:
        raise ValueError("Selected custom config was not found.")
    return str(src.get("yaml_text") or "")


def custom_slot_options(raw_slots: Any) -> list[dict[str, str]]:
    return [
        {
            "label": f"{item['name']} ({item.get('template_origin', 'custom')})",
            "value": str(item["slot_id"]),
        }
        for item in normalize_custom_slots(raw_slots)
    ]


def count_yaml_changes(left_yaml: str, right_yaml: str) -> int:
    try:
        left = _parse_yaml_mapping(left_yaml)
        right = _parse_yaml_mapping(right_yaml)
    except Exception:
        delta = difflib.ndiff(left_yaml.splitlines(), right_yaml.splitlines())
        return sum(1 for line in delta if line.startswith("+") or line.startswith("-"))

    def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                flatten(next_prefix, child, out)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value):
                next_prefix = f"{prefix}[{idx}]"
                flatten(next_prefix, child, out)
            if not value:
                out[prefix] = []
            return
        out[prefix] = value

    flat_left: dict[str, Any] = {}
    flat_right: dict[str, Any] = {}
    flatten("", left, flat_left)
    flatten("", right, flat_right)

    keys = set(flat_left.keys()) | set(flat_right.keys())
    return sum(1 for key in keys if flat_left.get(key) != flat_right.get(key))
