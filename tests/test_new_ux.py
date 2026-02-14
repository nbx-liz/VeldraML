"""Test GUI UX Flows - Simplified."""
import pytest
from unittest.mock import MagicMock, patch
from dash import html
import json

from veldra.gui.app import (
    _cb_inspect_data, 
    _cb_populate_builder_options,
    _cb_build_config_yaml,
    _cb_enqueue_run_job
)

@pytest.fixture
def mock_inspect_data():
    with patch("veldra.gui.app.inspect_data") as mock:
        yield mock

@pytest.fixture
def mock_submit_run():
    with patch("veldra.gui.app.submit_run_job") as mock:
        yield mock

def test_01_data_page_inspection_flow(mock_inspect_data):
    """Test Data Page inspection enabling the Next button."""
    mock_inspect_data.return_value = {
        "success": True, 
        "stats": {
            "n_rows": 100, 
            "n_cols": 2,
            "columns": ["col1", "col2"],
            "numeric_cols": ["col1"],
            "categorical_cols": ["col2"],
            "missing_count": 0
        }, 
        "preview": [{"col1": 1, "col2": "a"}]
    }
    
    # Callback args: n_clicks, upload_contents, upload_filename
    # Returns: (stats_div, error_msg, workflow_state, file_label, file_path)
    result = _cb_inspect_data(1, None, None)
    
    # Assertions
    assert result[0] is not None # stats_div
    assert result[1] == ""       # error_msg
    assert "data_path" in result[2]  # workflow state has data_path

    mock_inspect_data.return_value = {"success": False, "error": "File not found"}
    output_fail = _cb_inspect_data(1, None, None)
    assert output_fail[1] == "Error: File not found" # error_msg for failure
    assert output_fail[2] == {} # workflow_state should be empty on failure

def test_02_config_builder_population(mock_inspect_data):
    """Test Config Builder options population."""
    state = {"data_path": "data.csv", "target_col": "target"}
    mock_inspect_data.return_value = {"success": True, "stats": {"columns": ["A", "B", "target"]}}
    
    # Returns 8 elements (added target_opts at index 2)
    # Callback args: pathname, state
    res = _cb_populate_builder_options("/", state)
    d_path, t_col, target_opts = res[0], res[1], res[2]
    # res[5] is excluded cols options (non_target_opts)
    
    assert d_path == "data.csv"
    assert t_col == "target"
    # target_opts has all columns (A, B, target) -> 3
    assert len(target_opts) == 3 
    # res[5] excludes target (A, B) -> 2
    assert len(res[5]) == 2

def test_03_config_builder_yaml_generation():
    """Test YAML generation from Config Builder inputs."""
    yaml_out = _cb_build_config_yaml(
        "regression", "data.csv", "target", [], [], [],
        "random", 5, 42, None, None, None, None, None, None,
        0.1, 31, 100, -1, 20, 10, 1.0, 1.0, 0, 0,
        False, "fast", 10, "rmse", "artifacts",
        False, None, # causal
        None, None, None, None, None, None, None, None # search space
    )
    
    assert "type: regression" in yaml_out
    assert "path: data.csv" in yaml_out
    assert "target: target" in yaml_out
    assert "enabled: false" in yaml_out # tuning

def test_04_run_page_submission(mock_submit_run):
    """Test Run Page job submission."""
    mock_submit_run.return_value = MagicMock(message="Job started", job_id="job-123")
    
    # config_path_state fallback logic
    res = _cb_enqueue_run_job(
        1, "train", "yaml...", "configs/gui_run.yaml", "data.csv",
        "artifacts", None, "zip"
    )
    
    assert "[QUEUED]" in res
    assert "job-123" in res
    mock_submit_run.assert_called_once()
