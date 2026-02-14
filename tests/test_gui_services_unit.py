from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pandas as pd
from veldra.gui import services
from veldra.api.exceptions import VeldraValidationError
from veldra.gui.types import RunInvocation


def test_inspect_data(monkeypatch, tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")

    mock_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    monkeypatch.setattr(services, "load_tabular_data", lambda p: mock_df)

    res = services.inspect_data(str(data_path))
    assert res["success"] is True
    assert res["stats"]["n_rows"] == 2
    assert "a" in res["stats"]["numeric_cols"]

    # Test error
    res_err = services.inspect_data(str(tmp_path / "missing.csv"))
    assert res_err["success"] is False
    assert "not exist" in res_err["error"]

def test_runtime_management():
    class StoreStub:
        pass

    class WorkerStub:
        def __init__(self) -> None:
            self.started = 0
            self.stopped = 0

        def start(self) -> None:
            self.started += 1

        def stop(self) -> None:
            self.stopped += 1

    store = StoreStub()
    worker = WorkerStub()

    services.set_gui_runtime(job_store=store, worker=worker)
    assert services.get_gui_job_store() is store

    services._start_worker_if_needed()
    assert worker.started == 1

    services.stop_gui_runtime()
    assert worker.stopped == 1
    assert services._JOB_STORE is None

def test_run_action(monkeypatch):
    # Mock api functions
    monkeypatch.setattr(services, "fit", lambda c: "fit_res")
    monkeypatch.setattr(services, "tune", lambda c: "tune_res")
    monkeypatch.setattr(services, "estimate_dr", lambda c: "dr_res")
    monkeypatch.setattr(services, "evaluate", lambda c, d: "eval_res")
    monkeypatch.setattr(services, "simulate", lambda a, d, s: "sim_res")
    monkeypatch.setattr(services, "export", lambda a, format: "export_res")
    
    monkeypatch.setattr(services, "load_tabular_data", lambda p: pd.DataFrame())

    class ArtifactStub:
        @staticmethod
        def load(_path: str):
            return SimpleNamespace()

    monkeypatch.setattr(services, "Artifact", ArtifactStub)
    
    # Mock config resolution rules
    # _resolve_config calls load_config_yaml if path provided, or parses yaml string
    monkeypatch.setattr(services, "load_config_yaml", lambda p: "key: val")
    monkeypatch.setattr(services, "_load_config_from_yaml", lambda y: SimpleNamespace())
    
    # 1. Fit
    inv = RunInvocation("fit", config_yaml="key: val")
    res = services.run_action(inv)
    assert res.success, res.message
    assert res.payload["result"] == "fit_res"
    
    # 2. Unsupported
    inv_bad = RunInvocation("dance")
    res_bad = services.run_action(inv_bad)
    assert not res_bad.success
    assert "Unsupported" in res_bad.message
    
    # 3. Evaluate (Config)
    inv_eval = RunInvocation("evaluate", data_path="d", config_yaml="k: v")
    res_eval = services.run_action(inv_eval)
    assert res_eval.success
    
    # 4. Evaluate (Artifact)
    inv_eval_art = RunInvocation("evaluate", data_path="d", artifact_path="a")
    res_eval_art = services.run_action(inv_eval_art)
    assert res_eval_art.success
    
    # 5. Export
    inv_exp = RunInvocation("export", artifact_path="a", export_format="python")
    res_exp = services.run_action(inv_exp)
    assert res_exp.success
    
    # 6. Simulate
    # Needs scenarios path
    with patch("veldra.gui.services._load_scenarios", lambda _path: {"scenarios": []}):
        inv_sim = RunInvocation("simulate", artifact_path="a", data_path="d", scenarios_path="s")
        res_sim = services.run_action(inv_sim)
        assert res_sim.success

def test_list_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)

    run1 = root / "run1"
    run1.mkdir()
    (run1 / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "task_type": "reg", "created_at_utc": "2023-01-01"}),
        encoding="utf-8",
    )

    run2 = root / "run2"
    run2.mkdir()
    (root / "file.txt").write_text("x", encoding="utf-8")

    items = services.list_artifacts(str(root))
    assert len(items) == 1
    assert items[0].run_id == "r1"

def test_migration_services():
    # Test error paths
    with pytest.raises(VeldraValidationError):
        services.migrate_config_from_yaml("invalid: [yaml")
        
    with pytest.raises(VeldraValidationError):
        services.migrate_config_from_yaml("not: a dict\n- list") # yaml loads list

    with patch("veldra.gui.services.migrate_run_config_file", side_effect=Exception("Fail")):
        with pytest.raises(VeldraValidationError):
            services.migrate_config_file_via_gui(input_path="p")
