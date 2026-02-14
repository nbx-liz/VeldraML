from __future__ import annotations
from datetime import datetime, timezone

import pytest
import types
import dash
import plotly.graph_objs as go
import json
from dataclasses import dataclass
from veldra.gui.types import GuiJobRecord, RunInvocation

# Helper
def get_callback(app, output_substr: str):
    for key, value in app.callback_map.items():
        if output_substr in key:
            return value["callback"].__wrapped__
    raise KeyError(f"No callback found for output '{output_substr}'")

def test_app_coverage_edge_cases(monkeypatch):
    import veldra.gui.app as app_module
    app = app_module.create_app()
    
    # 1. _inspect_data coverage
    inspect_cb = get_callback(app, "data-inspection-result.children")
    
    # No upload (requires user file selection)
    res = inspect_cb(None, None, {})
    assert res[0] is None
    assert "Please select" in res[1]
    assert res[4] == ""
    
    # Inspection error
    monkeypatch.setattr(app_module, "inspect_data", lambda p: {"success": False, "error": "Fail"})
    res_err = inspect_cb("data:text/csv;base64,QQox", "x.csv", {})
    assert "Error: Fail" in res_err[1]
    assert len(res_err) == 5

    # 2. _save_target_col coverage
    # Find callback for workflow-state.data (allow_duplicate=True, so checkinputs)
    def get_save_target_callback(app):
        for key, value in app.callback_map.items():
            if "workflow-state.data" in key:
                 inputs = value["inputs"]
                 if any("data-target-col" in i["id"] for i in inputs):
                     return value["callback"].__wrapped__
        raise KeyError("Target Callback not found")
        
    save_cb = get_save_target_callback(app)
    # Empty state
    res_state = save_cb("target", None)
    assert res_state == {"target_col": "target"}
    
    # 3. _handle_migration_preview coverage
    preview_cb = get_callback(app, "config-migrate-diff.children")
    
    # File not found
    # Mock Path to return exists=False
    class MockPath:
        def __init__(self, p): pass
        def exists(self): return False
        def read_text(self, encoding="utf-8"): return ""
    
    monkeypatch.setattr(app_module, "Path", MockPath)
    res_prev = preview_cb(1, "bad_path", 1)
    # returns normalized, diff, error
    assert "File not found" in res_prev[2]
    
    # Exception
    monkeypatch.setattr(app_module, "Path", lambda p: (_ for _ in ()).throw(ValueError("Boom")))
    res_ex = preview_cb(1, "path", 1)
    assert "Error" in res_ex[2]
    
    # 4. _refresh_run_jobs Toast Logic coverage
    refresh_cb = get_callback(app, "run-jobs-table-container.children")
    
    # Mock jobs: ONE job that changed from queued to succeeded
    job_succeeded = GuiJobRecord("j1", "succeeded", "fit", "now", "now", RunInvocation("fit"), None)
    monkeypatch.setattr(app_module, "list_run_jobs", lambda limit=100: [job_succeeded])
    monkeypatch.setattr(app_module, "make_toast", lambda m, icon: f"Toast:{m}:{icon}")
    
    # Last status was queued
    last_status = {"j1": "queued"}
    
    _, toast, new_status, _, _ = refresh_cb(1, 0, last_status, {}, "/run", [])
    assert "Toast" in str(toast)
    assert "success" in str(toast)
    assert new_status["j1"] == "succeeded"

    # First-poll completion should not force-redirect from /run.
    now = datetime.now(timezone.utc).isoformat()
    job_first_poll = GuiJobRecord("jx", "succeeded", "fit", now, now, RunInvocation("fit"), None)
    monkeypatch.setattr(app_module, "list_run_jobs", lambda limit=100: [job_first_poll])
    _, _, _, _, next_path = refresh_cb(1, 0, {}, {}, "/run", [])
    assert next_path is dash.no_update
    
    # 5. _show_selected_job_detail coverage
    detail_cb = get_callback(app, "run-job-detail.children")
    
    # Index out of bounds
    res_oob = detail_cb([99], [{"job_id": "j1"}], None)
    assert "Job not found" in res_oob[0]
    
    # Job not found in store
    monkeypatch.setattr(app_module, "get_run_job", lambda jid: None)
    res_gone = detail_cb([0], [{"job_id": "j1"}], None)
    assert "unavailable" in res_gone[0]
    
    # Job with error
    job_err = GuiJobRecord("j2", "failed", "fit", "now", "now", RunInvocation("fit"), None, error_message="Fatal Error")
    monkeypatch.setattr(app_module, "get_run_job", lambda jid: job_err)
    res_err_msg = detail_cb([0], [{"job_id": "j2"}], None)
    # Returns html.Div, check string repr
    assert "Fatal Error" in str(res_err_msg[0])
    
    # Job with payload
    from veldra.gui.types import GuiRunResult
    res_obj = GuiRunResult(True, "Ok", {"acc": 0.9})
    job_payload = GuiJobRecord(
        "j3", "succeeded", "fit", "now", "now", 
        RunInvocation("fit"), 
        result=res_obj
    )
    monkeypatch.setattr(app_module, "get_run_job", lambda jid: job_payload)
    res_pay = detail_cb([0], [{"job_id": "j3"}], None)
    assert "acc" in str(res_pay[0])
    
    # 6. _cancel_job coverage
    # Find callback for run-result-log (allow_duplicate=True)
    def get_cancel_callback(app):
        for key, value in app.callback_map.items():
            if "run-result-log.children" in key:
                 inputs = value["inputs"]
                 if any("run-cancel-job-btn" in i["id"] for i in inputs):
                     return value["callback"].__wrapped__
        raise KeyError("Cancel callback not found")
        
    cancel_cb_real = get_cancel_callback(app)
    
    # Empty job id
    assert cancel_cb_real(1, None) == ""
    
    # Exception
    monkeypatch.setattr(app_module, "cancel_run_job", lambda jid: (_ for _ in ()).throw(ValueError("CancelFail")))
    assert "CancelFail" in cancel_cb_real(1, "j1")

    # 7. _list_artifacts Error coverage
    list_cb = get_callback(app, "artifact-select.options")
    monkeypatch.setattr(app_module, "list_artifacts", lambda p: (_ for _ in ()).throw(ValueError("ListFail")))
    res_list = list_cb(1, "/results", "root")
    assert res_list == ([], [])

    # 8. _update_result_view coverage
    view_cb = get_callback(app, "result-chart-main.figure")
    
    # No artifact path
    res_empty = view_cb(None, None)
    assert res_empty[0] == ""
    # The placeholder fig is a go.Figure, checking for layout content
    assert "layout" in res_empty[1]
    assert "annotations" in res_empty[1]["layout"]
    assert res_empty[3] == "Select an artifact to view results."
    class ArtifactStub:
        def __init__(self) -> None:
            self.metrics = {"r2_score": 0.5, "custom": 10}
            self.metadata = {}
            self.created_at_utc = "now"
            self.run_id = "r1"
            self.task_type = "reg"
            self.config = {}

    mock_art = ArtifactStub()
    
    def load_side_effect(path):
        if "bad" in path: raise ValueError("Bad")
        return mock_art
        
    class ArtifactLoader:
        @staticmethod
        def load(path):
            return load_side_effect(path)

    monkeypatch.setattr(app_module, "Artifact", ArtifactLoader)
    
    res_comp_err = view_cb("good", "bad")
    assert "Performance Metrics" in str(res_comp_err[1]) # generic title, check repr of figure object or just str
    # Since figure is Plotly object, str() works but might be large. Just check simpler part?
    # fig_main is returned as 2nd element.
    # It is likely a Figure object.
    
    # Extra metrics coverage
    assert "custom" in str(res_comp_err[0]) # KPI container children
    
    # 9. _evaluate_artifact_action coverage
    eval_cb = get_callback(app, "artifact-eval-result.children")
    
    # asdict failure fallback
    monkeypatch.setattr(app_module, "Artifact", ArtifactLoader)
    monkeypatch.setattr(app_module, "load_tabular_data", lambda p: "data")
    
    class NonDataclass:
        def __init__(self):
            # No dataclass fields
            pass
        @property
        def __dataclass_fields__(self):
            return {"score": str}
        @property
        def score(self):
            return 1.0
            
    monkeypatch.setattr(app_module, "evaluate", lambda a, d: NonDataclass())
    
    res_eval = eval_cb(1, "p", "d")
    assert '"score": 1.0' in res_eval
    
    # Large data handling (Lines 630+)
    @dataclass
    class ResLarge:
        data: dict
        
    monkeypatch.setattr(app_module, "evaluate", lambda a, d: ResLarge(data={"a": 1}))
    # This should pass through
    
    # "data" with to_dict
    @dataclass
    class ResMatrix:
        data: Any
        
    class Matrix:
        def to_dict(self): return {}
        
    monkeypatch.setattr(app_module, "evaluate", lambda a, d: ResMatrix(data=Matrix()))
    
    res_large = eval_cb(1, "p", "d")
    # ... (previous content)
    assert "data" not in res_large

    # 10. _handle_config_actions coverage
    config_cb = get_callback(app, "config-yaml.value")
    
    # Mock context for Import
    monkeypatch.setattr(app_module, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": "config-import-btn.n_clicks"}]))
    
    # No workflow state -> Error
    res_imp_err = config_cb(0, 0, 0, 1, "yaml", "path", None)
    assert "No data selected" in res_imp_err[1]
    
    # Import success
    state = {"data_path": "data.csv", "target_col": "tgt"}
    # Bad YAML input -> fallback to empty dict
    res_imp_ok = config_cb(0, 0, 0, 1, "{bad: yaml", "path", state)
    assert "Imported data settings" in res_imp_ok[1]
    assert "data.csv" in res_imp_ok[0]
    
    # Validate
    monkeypatch.setattr(app_module, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": "config-validate-btn.n_clicks"}]))
    monkeypatch.setattr(app_module, "validate_config", lambda y: True)
    res_val = config_cb(1, 0, 0, 0, "yaml", "path", state)
    assert "valid" in res_val[1]
    
    # Exception in config action
    monkeypatch.setattr(app_module, "validate_config", lambda y: (_ for _ in ()).throw(ValueError("ConfigFail")))
    res_conf_ex = config_cb(1, 0, 0, 0, "yaml", "path", state)
    assert "ConfigFail" in res_conf_ex[1]

    # 11. _handle_migration_apply
    apply_cb = get_callback(app, "config-migrate-result.children") # allow_duplicate=True
    # Need to distinguish from Preview callback?
    # Preview outputs (normalized, diff, result).
    # Apply outputs (result).
    # So `config-migrate-result.children` appears in BOTH.
    # One is 3rd output of Preview. One is 1st output of Apply.
    # get_callback iterates.
    
    def get_apply_callback(app):
        for key, value in app.callback_map.items():
            if "config-migrate-result.children" in key:
                 # Check inputs
                 inputs = value["inputs"]
                 if any("config-migrate-apply-btn" in i["id"] for i in inputs):
                     return value["callback"].__wrapped__
        raise KeyError("Apply callback not found")
        
    apply_cb_real = get_apply_callback(app)
    
    # Mock toast
    monkeypatch.setattr(app_module, "make_toast", lambda m, icon="success": f"Toast:{m}")

    monkeypatch.setattr(app_module, "migrate_config_file_via_gui", lambda p, target_version=1: "Migrated!")
    res_app = apply_cb_real(1, "path", 1)
    assert "Migrated!" in str(res_app)
    
    monkeypatch.setattr(app_module, "migrate_config_file_via_gui", lambda p, target_version=1: (_ for _ in ()).throw(ValueError("MigFail")))
    res_app_err = apply_cb_real(1, "path", 1)
    assert "MigFail" in str(res_app_err)

    # 12. _set_run_polling
    poll_cb = get_callback(app, "run-jobs-interval.interval")
    assert poll_cb("/") >= 200
