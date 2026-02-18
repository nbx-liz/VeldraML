from __future__ import annotations

from veldra.gui.template_service import clone_custom_slot, normalize_custom_slots, save_custom_slot


def test_custom_slot_limit_and_lru() -> None:
    slots: list[dict] = []
    for idx in range(12):
        slots = save_custom_slot(
            slots,
            name=f"cfg-{idx}",
            yaml_text=f"config_version: 1\nidx: {idx}\n",
            template_origin="custom",
        )
    normalized = normalize_custom_slots(slots)
    assert len(normalized) == 10
    names = {item["name"] for item in normalized}
    assert "cfg-11" in names
    assert "cfg-0" not in names


def test_clone_slot_generates_copy_name() -> None:
    slots: list[dict] = []
    slots = save_custom_slot(
        slots,
        name="baseline",
        yaml_text="config_version: 1\n",
        template_origin="builtin",
    )
    slot_id = slots[0]["slot_id"]
    cloned = clone_custom_slot(slots, slot_id=slot_id)
    names = [item["name"] for item in cloned]
    assert any(name.startswith("baseline (copy") for name in names)
