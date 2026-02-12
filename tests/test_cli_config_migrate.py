from __future__ import annotations

import json
from pathlib import Path

import pytest

from veldra import main


def _write_valid_config(path: Path) -> None:
    path.write_text(
        "config_version: 1\n"
        "task:\n"
        "  type: regression\n"
        "data:\n"
        "  path: train.csv\n"
        "  target: y\n",
        encoding="utf-8",
    )


def test_cli_config_migrate_success(tmp_path, capsys) -> None:
    src = tmp_path / "run.yaml"
    _write_valid_config(src)
    main(["config", "migrate", "--input", str(src)])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["source_version"] == 1
    assert payload["target_version"] == 1
    assert payload["output_path"].endswith("run.migrated.yaml")


def test_cli_config_migrate_requires_input() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["config", "migrate"])
    assert exc.value.code == 2


def test_cli_config_migrate_validation_error(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "config_version: 1\n"
        "task:\n"
        "  type: regression\n"
        "data:\n"
        "  path: train.csv\n"
        "  target: y\n"
        "unknown_key: true\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["config", "migrate", "--input", str(bad)])
    assert "ERROR:" in str(exc.value)
