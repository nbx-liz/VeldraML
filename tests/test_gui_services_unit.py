from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
from veldra.gui import services
from veldra.api.exceptions import VeldraValidationError
from veldra.gui.types import RunInvocation

@pytest.fixture
def mock_path(monkeypatch):
    mock = MagicMock(spec=Path)
    # Monkeypatching Path in services.py is hard if it imports names.
    # services.py imports Path.
    # We might need to patch 'veldra.gui.services.Path'
    return mock

def test_inspect_data(monkeypatch):
    # Mock specific Path usage inside inspect_data
    # It calls Path(path), .exists(), load_tabular_data, df ops.
    
    mock_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    monkeypatch.setattr(services, "load_tabular_data", lambda p: mock_df)
    
    with patch("veldra.gui.services.Path") as MockPath:
        MockPath.return_value.exists.return_value = True
        MockPath.return_value.resolve.return_value = "abs/path"
        
        res = services.inspect_data("data.csv")
        assert res["success"] is True
        assert res["stats"]["n_rows"] == 2
        assert "a" in res["stats"]["numeric_cols"]
        
        # Test error
        MockPath.return_value.exists.return_value = False
        res_err = services.inspect_data("data.csv")
        assert res_err["success"] is False
        assert "not exist" in res_err["error"]

def test_runtime_management():
    store = MagicMock()
    worker = MagicMock()
    
    services.set_gui_runtime(job_store=store, worker=worker)
    assert services.get_gui_job_store() is store
    
    services._start_worker_if_needed()
    worker.start.assert_called_once()
    
    services.stop_gui_runtime()
    worker.stop.assert_called_once()
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
    monkeypatch.setattr(services, "Artifact", MagicMock())
    monkeypatch.setattr(services.Artifact, "load", lambda p: MagicMock())
    
    # Mock config resolution rules
    # _resolve_config calls load_config_yaml if path provided, or parses yaml string
    monkeypatch.setattr(services, "load_config_yaml", lambda p: "key: val")
    monkeypatch.setattr(services, "_load_config_from_yaml", lambda y: MagicMock())
    
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
    with patch("veldra.gui.services.Path") as MP:
        MP.return_value.exists.return_value = True
        MP.return_value.read_text.return_value = '{"scenarios": []}'
        MP.return_value.suffix = ".json"
        
        inv_sim = RunInvocation("simulate", artifact_path="a", data_path="d", scenarios_path="s")
        res_sim = services.run_action(inv_sim)
        assert res_sim.success

def test_list_artifacts():
    with patch("veldra.gui.services.Path") as MockPath:
        root = MockPath.return_value
        root.exists.return_value = True
        root.is_dir.return_value = True
        
        # Child 1: Valid
        c1 = MagicMock()
        c1.is_dir.return_value = True
        c1.name = "run1"
        c1.__lt__ = lambda self, other: self.name < other.name
        c1.__truediv__.return_value.exists.return_value = True # manifest exists
        c1.__truediv__.return_value.read_text.return_value = json.dumps({
            "run_id": "r1", "task_type": "reg", "created_at_utc": "2023-01-01"
        })
        
        # Child 2: No manifest
        c2 = MagicMock()
        c2.is_dir.return_value = True
        c2.name = "run2"
        c2.__lt__ = lambda self, other: self.name < other.name
        c2.__truediv__.return_value.exists.return_value = False
        
        # Child 3: File
        c3 = MagicMock()
        c3.is_dir.return_value = False
        # c3 might be sorted if is_dir checked inside loop.
        # sorted(root.iterdir()) happens BEFORE loop.
        c3.__lt__ = lambda self, other: self.name < other.name
        c3.name = "file"
        
        root.iterdir.return_value = [c1, c2, c3]
        
        items = services.list_artifacts("root")
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
