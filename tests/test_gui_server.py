from __future__ import annotations

import sys
import types

import pytest

from veldra.gui import server


def test_build_parser_uses_environment_defaults(monkeypatch) -> None:
    monkeypatch.setenv("VELDRA_GUI_HOST", "0.0.0.0")
    monkeypatch.setenv("VELDRA_GUI_PORT", "9000")
    monkeypatch.setenv("VELDRA_GUI_DEBUG", "1")
    parser = server._build_parser()
    args = parser.parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.debug is True


def test_main_runs_dash_app(monkeypatch) -> None:
    called: dict[str, object] = {}

    class _FakeApp:
        def run(self, host: str, port: int, debug: bool) -> None:
            called["host"] = host
            called["port"] = port
            called["debug"] = debug

    fake_mod = types.ModuleType("veldra.gui.app")
    fake_mod.create_app = lambda: _FakeApp()
    monkeypatch.setitem(sys.modules, "veldra.gui.app", fake_mod)
    monkeypatch.setattr(
        server,
        "log_event",
        lambda *args, **kwargs: called.setdefault("logged", True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["veldra-gui", "--host", "127.0.0.2", "--port", "8123", "--debug"],
    )
    server.main()

    assert called["host"] == "127.0.0.2"
    assert called["port"] == 8123
    assert called["debug"] is True
    assert called["logged"] is True


def test_main_raises_runtime_error_when_gui_deps_missing(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["veldra-gui"])
    monkeypatch.delitem(sys.modules, "veldra.gui.app", raising=False)
    real_import = __import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "veldra.gui.app":
            raise ModuleNotFoundError("dash not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    with pytest.raises(RuntimeError, match="GUI dependencies are not installed"):
        server.main()
