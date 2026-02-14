from __future__ import annotations

from unittest.mock import MagicMock

from veldra.gui.app import _json_dumps


def test_json_dumps_handles_magicmock_without_recursive_blowup() -> None:
    cfg = MagicMock()
    cfg.model_dump.return_value = MagicMock()
    payload = {"run_config": cfg}
    out = _json_dumps(payload)
    assert "run_config" in out


def test_json_dumps_handles_cycle() -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload
    out = _json_dumps(payload)
    assert "cycle" in out
