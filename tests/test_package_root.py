from __future__ import annotations

from veldra import main


def test_package_main_shows_help(capsys) -> None:
    main([])
    out = capsys.readouterr().out
    assert "Veldra command line tools." in out

