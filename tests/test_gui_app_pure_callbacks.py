from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import dash
import pandas as pd

from veldra.gui import app as app_module


def test_update_run_launch_state_table_driven() -> None:
    cases = [
        ("fit", "", "", "", "", "", True, "Data Source, Config Source"),
        ("fit", "d.csv", "x", "", "", "", False, "Ready"),
        ("tune", "d.csv", "", "cfg.yml", "", "", False, "Ready"),
        ("estimate_dr", "", "cfg", "", "", "", True, "Data Source"),
        ("evaluate", "d.csv", "", "", "", "", True, "Artifact Path or Config Source"),
        ("evaluate", "d.csv", "", "", "art", "", False, "Ready"),
        ("simulate", "d.csv", "", "", "", "", True, "Artifact Path, Scenarios Path"),
        ("simulate", "d.csv", "", "", "art", "sc.yml", False, "Ready"),
        ("export", "", "", "", "", "", True, "Artifact Path"),
        ("export", "", "", "", "art", "", False, "Ready"),
    ]
    for action, data, cfg, cfg_path, art, scn, disabled, expected in cases:
        out = app_module._cb_update_run_launch_state(action, data, cfg, cfg_path, art, scn)
        assert out[0] is disabled
        assert expected in out[1]


def test_autoselect_artifact_conditions() -> None:
    options = [{"label": "new", "value": "a/new"}, {"label": "old", "value": "a/old"}]

    assert app_module._cb_autoselect_artifact({}, options, "/run", None) is dash.no_update
    assert app_module._cb_autoselect_artifact({}, [], "/results", None) is dash.no_update
    assert (
        app_module._cb_autoselect_artifact(
            {"last_run_artifact": "a/old"}, options, "/results", "a/new"
        )
        == "a/old"
    )
    assert app_module._cb_autoselect_artifact({}, options, "/results", "a/new") is dash.no_update
    assert app_module._cb_autoselect_artifact({}, options, "/results", "missing") == "a/new"


def test_to_jsonable_handles_cycles_mock_depth_and_model_dump_errors() -> None:
    cyc: dict[str, object] = {}
    cyc["self"] = cyc
    out = app_module._to_jsonable(cyc)
    assert out["self"] == "<cycle>"

    deep = []
    cur = deep
    for _ in range(30):
        nxt = []
        cur.append(nxt)
        cur = nxt
    rendered = app_module._to_jsonable(deep)
    assert "<max_depth_reached>" in str(rendered)

    assert "Mock" in app_module._to_jsonable(Mock())

    class _BrokenDump:
        def model_dump(self, mode: str = "json"):  # noqa: ARG002
            raise RuntimeError("boom")

    assert "BrokenDump" in app_module._to_jsonable(_BrokenDump())


def test_json_dumps_and_sync_path_preset(monkeypatch) -> None:
    text = app_module._json_dumps({"x": 1})
    assert '"x": 1' in text

    monkeypatch.setattr(app_module, "callback_context", SimpleNamespace(triggered_id="preset"))
    assert app_module._sync_path_preset("custom", "any")[0] == "custom"
    assert app_module._sync_path_preset("artifacts", "any") == ("artifacts", "artifacts")

    monkeypatch.setattr(app_module, "callback_context", SimpleNamespace(triggered_id="input"))
    assert app_module._sync_path_preset("custom", "artifacts") == ("artifacts", "artifacts")
    assert app_module._sync_path_preset("custom", "output") == ("output", "output")
    assert app_module._sync_path_preset("custom", "x") == ("custom", "x")


def test_evaluate_artifact_action_fallbacks(monkeypatch) -> None:
    @dataclass
    class _EvalResult:
        metrics: dict[str, float]
        data: pd.DataFrame

    monkeypatch.setattr(app_module, "Artifact", SimpleNamespace(load=lambda _p: object()))
    monkeypatch.setattr(app_module, "_get_load_tabular_data", lambda: (lambda _p: pd.DataFrame()))
    monkeypatch.setattr(
        app_module,
        "_get_evaluate",
        lambda: (lambda _a, _f: _EvalResult(metrics={"acc": 0.9}, data=pd.DataFrame({"x": [1]}))),
    )
    ok = app_module._cb_evaluate_artifact_action(1, "art", "data")
    assert '"metrics"' in ok
    assert "DataFrame" in ok

    class _HasDict:
        def to_dict(self):
            return {}

    @dataclass
    class _DataObj:
        data: object
        v: int = 1

    monkeypatch.setattr(
        app_module,
        "_get_evaluate",
        lambda: (lambda _a, _f: _DataObj(data=_HasDict())),
    )
    out = app_module._cb_evaluate_artifact_action(1, "art", "data")
    assert '"data"' not in out

    assert "required" in app_module._cb_evaluate_artifact_action(1, "", "")
