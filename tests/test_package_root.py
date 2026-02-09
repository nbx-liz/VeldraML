from __future__ import annotations

from veldra import main


def test_package_main_prints_scaffold_message(capsys) -> None:
    main()
    out = capsys.readouterr().out
    assert "VeldraML scaffold is ready" in out

