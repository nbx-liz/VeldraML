from __future__ import annotations
import types
from unittest.mock import MagicMock
import pytest
from dash import html

def get_callback(app, output_substr: str):
    for key, value in app.callback_map.items():
        if output_substr in key:
            return value["callback"].__wrapped__
    raise KeyError(f"No callback found for output '{output_substr}'")

def test_config_actions(monkeypatch):
    import veldra.gui.app as app_module
    app = app_module.create_app()
    
    # Config Actions: Load, Save, Import, Validate
    # Output: config-yaml.value...
    action_cb = get_callback(app, "config-yaml.value")
    
    # Mock helpers
    monkeypatch.setattr(app_module, "make_toast", lambda m, **k: "toast")
    monkeypatch.setattr(app_module, "validate_config", lambda y: True)
    
    # 1. Config Load (already covered partly in internal tests, but let's reinforce)
    monkeypatch.setattr(
        app_module,
        "callback_context",
        types.SimpleNamespace(triggered=[{"prop_id": "config-load-btn.n_clicks"}])
    )
    monkeypatch.setattr(app_module, "load_config_yaml", lambda p: "loaded_yml")
    
    res_load = action_cb(1, 0, 0, 0, "curr", "path", {})
    assert res_load[0] == "loaded_yml"
    assert "Loaded" in res_load[1]
    
    # 2. Config Save
    monkeypatch.setattr(
        app_module,
        "callback_context",
        types.SimpleNamespace(triggered=[{"prop_id": "config-save-btn.n_clicks"}])
    )
    monkeypatch.setattr(app_module, "save_config_yaml", lambda p, y: "saved_path")
    
    res_save = action_cb(0, 1, 0, 0, "yml", "path", {})
    assert res_save[1] == "Saved: saved_path"
    
    # 3. Config Import
    monkeypatch.setattr(
        app_module,
        "callback_context",
        types.SimpleNamespace(triggered=[{"prop_id": "config-import-btn.n_clicks"}])
    )
    
    # Case: No workflow state
    res_imp_err = action_cb(0, 0, 1, 0, "yml", "path", None)
    assert "No data selected" in res_imp_err[1]
    assert res_imp_err[2]["color"] == "#ef4444" # Error style
    
    # Case: Valid state
    state = {"data_path": "data.csv", "target_col": "target"}
    res_imp = action_cb(0, 0, 1, 0, "task: regression", "path", state)
    assert "Imported data settings" in res_imp[1]
    assert "data.csv" in res_imp[0]
    assert "target" in res_imp[0]
    
    # Case: Invalid YAML input
    res_imp_bad = action_cb(0, 0, 1, 0, "{bad_yaml", "path", state)
    assert "Imported" in res_imp_bad[1] # Should recover and use empty dict
    
    # 4. Validate (Default fallthrough)
    monkeypatch.setattr(
        app_module,
        "callback_context",
        types.SimpleNamespace(triggered=[{"prop_id": "config-yaml.value"}])
    )
    res_val = action_cb(0, 0, 0, 0, "yml", "path", {})
    assert "valid" in res_val[1]
    
    # Error in validation
    monkeypatch.setattr(app_module, "validate_config", lambda y: (_ for _ in ()).throw(ValueError("Invalid")))
    res_val_err = action_cb(0, 0, 0, 0, "yml", "path", {})
    assert "Invalid" in res_val_err[1]

def test_migration_apply(monkeypatch):
    import veldra.gui.app as app_module
    app = app_module.create_app()
    
    # Migration Apply
    # Output: config-migrate-result.children (dup)
    # We check allow_duplicate=True
    # Keys might be suffixed.
    # We loop to find the one with Input("config-migrate-apply-btn")
    
    apply_cb = None
    for val in app.callback_map.values():
        cb = val["callback"]
        # In Dash < 2.0 structure might differ, but usually 'inputs' is list of dicts/objects
        # str(inputs) works if it contains id.
        # Let's inspect inputs directly if possible
        inputs = val["inputs"]
        # inputs can be list of dicts [{'id': '...', 'property': '...'}]
        found = False
        for i in inputs:
            if hasattr(i, "component_id") and i.component_id == "config-migrate-apply-btn":
                found = True
                break
            if isinstance(i, dict) and i.get("id") == "config-migrate-apply-btn":
                found = True
                break
            # Fallback string check
            if "config-migrate-apply-btn" in str(i):
                found = True
                break
        
        if found:
            apply_cb = cb.__wrapped__ if hasattr(cb, "__wrapped__") else cb
            break
            
    assert apply_cb is not None
    
    monkeypatch.setattr(app_module, "migrate_config_file_via_gui", lambda p, target_version=1: "Success")
    
    res = apply_cb(1, "path", 1)
    assert "Success" in str(res) # html.Div(html.Pre("Success")) -> str usually shows structure or we can inspect children
    
    # Error
    monkeypatch.setattr(app_module, "migrate_config_file_via_gui", lambda p, target_version=1: (_ for _ in ()).throw(ValueError("Fail")))
    res_err = apply_cb(1, "path", 1)
    assert "Fail" in str(res_err)
