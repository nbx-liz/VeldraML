from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.api.types import SimulationResult
from veldra.gui.services import (
    _load_scenarios,
    _require,
    _resolve_config,
    _result_to_payload,
    load_config_yaml,
    run_action,
)
from veldra.gui.types import RunInvocation


def test_require_validation() -> None:
    with pytest.raises(VeldraValidationError):
        _require("", "path")
    with pytest.raises(VeldraValidationError):
        _require("   ", "path")
    assert _require(" x ", "path") == "x"


def test_load_config_yaml_missing_file(tmp_path) -> None:
    with pytest.raises(VeldraValidationError):
        load_config_yaml(str(tmp_path / "missing.yaml"))


def test_result_to_payload_variants() -> None:
    @dataclass
    class _DataclassResult:
        data: pd.DataFrame

    payload = _result_to_payload(pd.DataFrame({"x": [1, 2]}))
    assert payload["n_rows"] == 2

    payload2 = _result_to_payload({"a": 1})
    assert payload2 == {"result": {"a": 1}}

    payload3 = _result_to_payload(None)
    assert payload3 == {"result": None}

    payload4 = _result_to_payload(_DataclassResult(data=pd.DataFrame({"y": [3]})))
    assert payload4["data"]["n_rows"] == 1

    class _Unknown:
        pass

    payload5 = _result_to_payload(_Unknown())
    assert "result_repr" in payload5


def test_resolve_config_from_path(tmp_path) -> None:
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "config_version: 1\ntask:\n  type: regression\ndata:\n  path: train.csv\n  target: y\n",
        encoding="utf-8",
    )
    resolved = _resolve_config(RunInvocation(action="fit", config_path=str(cfg)))
    assert resolved.task.type == "regression"


def test_load_scenarios_json_and_yaml(tmp_path) -> None:
    js = tmp_path / "s.json"
    js.write_text(json.dumps({"name": "s1", "actions": []}), encoding="utf-8")
    ya = tmp_path / "s.yaml"
    ya.write_text("name: s2\nactions: []\n", encoding="utf-8")
    assert _load_scenarios(str(js))["name"] == "s1"
    assert _load_scenarios(str(ya))["name"] == "s2"
    with pytest.raises(VeldraValidationError):
        _load_scenarios(str(tmp_path / "none.yaml"))


def test_run_action_errors(monkeypatch) -> None:
    result = run_action(RunInvocation(action="unknown"))
    assert result.success is False
    assert "Unsupported action" in result.message

    # evaluate path with config branch
    frame = pd.DataFrame({"x": [1.0], "y": [2.0]})
    monkeypatch.setattr("veldra.gui.services.load_tabular_data", lambda _p: frame)
    monkeypatch.setattr(
        "veldra.gui.services.evaluate",
        lambda _cfg, _f: SimulationResult(task_type="r", data=frame),
    )
    cfg_yaml = (
        "config_version: 1\ntask:\n  type: regression\ndata:\n  path: train.csv\n  target: y\n"
    )
    ok = run_action(RunInvocation(action="evaluate", config_yaml=cfg_yaml, data_path="eval.csv"))
    assert ok.success is True
