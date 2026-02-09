from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from examples import common


@dataclass
class _Payload:
    name: str
    values: list[int]


def test_make_timestamp_dir_uses_suffix_when_name_collides(tmp_path, monkeypatch) -> None:
    class _Now:
        @staticmethod
        def strftime(_: str) -> str:
            return "20260210_010203"

    class _DatetimeMock:
        @staticmethod
        def now(_: object) -> _Now:
            return _Now()

    monkeypatch.setattr(common, "datetime", _DatetimeMock)
    first = tmp_path / "20260210_010203"
    first.mkdir()

    created = common.make_timestamp_dir(tmp_path)

    assert created.name == "20260210_010203_01"
    assert created.exists()


def test_to_jsonable_handles_dataclass_path_dict_list_and_numpy() -> None:
    payload = {
        "d": _Payload(name="x", values=[1, 2]),
        "p": Path("abc/def.txt"),
        "arr": np.array([3, 4]),
    }

    out = common.to_jsonable(payload)

    assert out["d"] == {"name": "x", "values": [1, 2]}
    assert str(out["p"]).endswith("def.txt")
    assert out["arr"] == [3, 4]


def test_save_json_and_yaml_write_expected_content(tmp_path) -> None:
    json_path = tmp_path / "a" / "b.json"
    yaml_path = tmp_path / "a" / "b.yaml"

    common.save_json(json_path, {"z": 1, "a": [2, 3]})
    common.save_yaml(yaml_path, {"k": "v", "n": 2})

    assert json_path.exists()
    assert yaml_path.exists()
    assert '"z": 1' in json_path.read_text(encoding="utf-8")
    loaded_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert loaded_yaml == {"k": "v", "n": 2}


def test_format_error_includes_hint() -> None:
    text = common.format_error(ValueError("boom"), "retry")
    assert "boom" in text
    assert "Hint: retry" in text
