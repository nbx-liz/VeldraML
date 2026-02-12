from __future__ import annotations

import pandas as pd

from veldra.api.types import (
    CausalResult,
    EvalResult,
    ExportResult,
    RunResult,
    SimulationResult,
    TuneResult,
)
from veldra.gui.job_store import GuiJobStore
from veldra.gui.services import run_action, set_gui_runtime, stop_gui_runtime, submit_run_job
from veldra.gui.types import RunInvocation


def test_run_action_dispatch_for_config_actions(monkeypatch) -> None:
    called: dict[str, int] = {"fit": 0, "tune": 0, "estimate_dr": 0}

    def _fake_fit(config):
        called["fit"] += 1
        return RunResult(run_id="r1", task_type=config.task.type, artifact_path="artifacts/r1")

    def _fake_tune(config):
        called["tune"] += 1
        return TuneResult(run_id="r2", task_type=config.task.type, best_score=0.1)

    def _fake_estimate(config):
        called["estimate_dr"] += 1
        return CausalResult(run_id="r3", method="dr", estimand="att", estimate=1.0)

    monkeypatch.setattr("veldra.gui.services.fit", _fake_fit)
    monkeypatch.setattr("veldra.gui.services.tune", _fake_tune)
    monkeypatch.setattr("veldra.gui.services.estimate_dr", _fake_estimate)

    yaml_text = """
config_version: 1
task:
  type: regression
data:
  path: train.csv
  target: y
split:
  type: kfold
  n_splits: 2
  seed: 7
export:
  artifact_dir: artifacts
    """.strip()

    for action in ("fit", "tune", "estimate_dr"):
        result = run_action(RunInvocation(action=action, config_yaml=yaml_text))
        assert result.success is True

    assert called == {"fit": 1, "tune": 1, "estimate_dr": 1}


def test_run_action_dispatch_for_artifact_actions(monkeypatch) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 1.0]})
    fake_artifact = object()
    called: dict[str, int] = {"evaluate": 0, "simulate": 0, "export": 0}

    monkeypatch.setattr("veldra.gui.services.Artifact.load", lambda _path: fake_artifact)
    monkeypatch.setattr("veldra.gui.services.load_tabular_data", lambda _path: frame)
    monkeypatch.setattr(
        "veldra.gui.services._load_scenarios",
        lambda _path: {"name": "s1", "actions": [{"op": "add", "column": "x", "value": 1.0}]},
    )

    def _fake_eval(artifact, data):
        assert artifact is fake_artifact
        assert data is frame
        called["evaluate"] += 1
        return EvalResult(task_type="regression", metrics={"rmse": 1.2})

    def _fake_sim(artifact, data, scenarios):
        assert artifact is fake_artifact
        assert data is frame
        assert isinstance(scenarios, dict)
        called["simulate"] += 1
        return SimulationResult(task_type="regression", data=data)

    def _fake_export(artifact, format):
        assert artifact is fake_artifact
        assert format == "python"
        called["export"] += 1
        return ExportResult(path="artifacts/exports/r1/python", format="python")

    monkeypatch.setattr("veldra.gui.services.evaluate", _fake_eval)
    monkeypatch.setattr("veldra.gui.services.simulate", _fake_sim)
    monkeypatch.setattr("veldra.gui.services.export", _fake_export)

    eval_result = run_action(
        RunInvocation(action="evaluate", artifact_path="artifacts/r1", data_path="eval.csv")
    )
    sim_result = run_action(
        RunInvocation(
            action="simulate",
            artifact_path="artifacts/r1",
            data_path="eval.csv",
            scenarios_path="scenarios.yaml",
        )
    )
    export_result = run_action(
        RunInvocation(action="export", artifact_path="artifacts/r1", export_format="python")
    )

    assert eval_result.success is True
    assert sim_result.success is True
    assert export_result.success is True
    assert called == {"evaluate": 1, "simulate": 1, "export": 1}


def test_submit_run_job_dispatches_to_queue(tmp_path) -> None:
    class _Worker:
        def __init__(self) -> None:
            self.started = 0

        def start(self) -> None:
            self.started += 1

    store = GuiJobStore(tmp_path / "jobs.sqlite3")
    worker = _Worker()
    set_gui_runtime(job_store=store, worker=worker)
    queued = submit_run_job(RunInvocation(action="fit"))
    assert queued.status == "queued"
    assert worker.started == 1
    stop_gui_runtime()
