import base64
from dataclasses import dataclass
from unittest.mock import mock_open, patch

import pandas as pd

# Import the callbacks to test
from veldra.gui.app import (
    _cb_build_config_yaml,
    _cb_detect_run_action,
    _cb_evaluate_artifact_action,
    _cb_inspect_data,
    _cb_update_result_view,
    _cb_update_selected_file_label,
    _cb_update_split_options,
    _cb_update_tune_objectives,
    _cb_update_tune_visibility,
    _sync_path_preset,
)

# --- 1. _cb_inspect_data (Upload) ---


@patch("veldra.gui.app.inspect_data")
@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
def test_inspect_data_upload_csv(mock_makedirs, mock_open, mock_inspect):
    """Test CSV upload handling."""
    # Mock inspect_data success
    mock_inspect.return_value = {
        "success": True,
        "stats": {
            "columns": ["A"],
            "n_rows": 10,
            "n_cols": 1,
            "numeric_cols": [],
            "categorical_cols": [],
            "missing_count": 0,
        },
        "preview": [{"A": 1}],
    }

    # Create valid base64 CSV content
    content_string = base64.b64encode(b"A\n1").decode("utf-8")
    upload_contents = f"data:text/csv;base64,{content_string}"

    res = _cb_inspect_data(1, upload_contents, "new_data.csv")

    assert res[0] is not None  # stats div
    assert res[1] == ""  # no error
    assert "temp_data" in res[2]["data_path"]
    assert "new_data.csv" in res[2]["data_path"]
    assert "new_data.csv" in res[3]  # file label]


@patch("veldra.gui.app.inspect_data")
def test_inspect_data_upload_unsupported(mock_inspect):
    """Test unsupported file type."""
    # Use valid base64 (MTIz = 123)
    upload_contents = "data:text/plain;base64,MTIz"
    res = _cb_inspect_data(1, upload_contents, "data.txt")
    assert "Unsupported file type" in res[1]


def test_update_selected_file_label():
    assert _cb_update_selected_file_label(None)[0].startswith("No file selected")
    assert "sample.csv" in _cb_update_selected_file_label("sample.csv")[0]


# --- 2. _cb_detect_run_action ---


def test_detect_run_action():
    # Default/Empty
    assert _cb_detect_run_action(None)[0] == "fit"
    assert "TRAIN" in _cb_detect_run_action("")[1]

    # Tune
    yaml_tune = "tuning:\n  enabled: true"
    assert _cb_detect_run_action(yaml_tune)[0] == "tune"
    assert "TUNE" in _cb_detect_run_action(yaml_tune)[1]

    # Causal
    yaml_causal = "task:\n  causal_method: dr"
    assert _cb_detect_run_action(yaml_causal)[0] == "fit"
    assert "CAUSAL" in _cb_detect_run_action(yaml_causal)[1]

    # Malformed
    assert _cb_detect_run_action("invalid: [")[0] == "fit"


# --- 3. _cb_build_config_yaml (Search Space & Splits) ---


def test_build_config_yaml_full():
    """Test comprehensive YAML generation."""
    yaml_str = _cb_build_config_yaml(
        "regression",
        "data.csv",
        "target",
        ["id"],
        ["cat"],
        ["drop"],
        "timeseries",
        5,
        42,
        "group_col",
        "time_col",
        "rolling",
        0.2,
        1,
        1,  # split params
        0.1,
        31,
        100,
        -1,
        20,
        10,
        1.0,
        1.0,
        0,
        0,  # train params
        True,
        "custom",
        20,
        "rmse",
        "artifacts",  # tuning
        True,
        "dr",  # causal
        "0.01",
        "0.1",
        "10",
        "50",
        "3",
        "10",
        "0.5",
        "1.0",  # search space
    )

    assert "causal_method: dr" in yaml_str
    assert "search_space:" in yaml_str
    assert "learning_rate:" in yaml_str
    assert "timeseries_mode: rolling" in yaml_str
    assert "gap: 1" in yaml_str


# --- 4. _cb_update_result_view ---


def test_result_view_empty_and_error():
    # Empty path
    res = _cb_update_result_view(None, None)
    assert res[0] == ""
    assert "Select an artifact" in res[3]

    # Error loading
    with patch("veldra.api.artifact.Artifact.load", side_effect=Exception("Load fail")):
        res = _cb_update_result_view("bad_path", None)
        assert "Error loading artifact" in str(res[0])


# --- 5. _cb_evaluate_artifact_action ---


@patch("veldra.gui.app.evaluate")
@patch("veldra.gui.app.load_tabular_data")
@patch("veldra.gui.app.Artifact.load")
def test_evaluate_action(mock_load, mock_data, mock_eval):
    # Happy path
    @dataclass
    class MockResult:
        metrics: dict
        preds: pd.DataFrame

    mock_res = MockResult(metrics={"acc": 0.9}, preds=pd.DataFrame({"a": [1]}))
    mock_eval.return_value = mock_res

    res = _cb_evaluate_artifact_action(1, "art_path", "data_path")
    assert '"metrics":' in res
    assert "DataFrame" in res  # Checks dataframe serialization logic

    # Error path
    mock_eval.side_effect = Exception("Eval fail")
    res_err = _cb_evaluate_artifact_action(1, "art_path", "data_path")
    assert "Evaluation failed" in res_err


# --- 6. Minor Callbacks ---


@patch("veldra.gui.app.callback_context")
def test_minor_callbacks(mock_ctx):
    # Sync path preset - Preset changed
    mock_ctx.triggered_id = "preset"
    assert _sync_path_preset("custom", "any")[0] == "custom"

    # Sync path preset - Input changed
    mock_ctx.triggered_id = "input"
    assert _sync_path_preset("custom", "artifacts")[0] == "artifacts"

    # Update Tune Visibility
    assert _cb_update_tune_visibility(True) == {"display": "block"}
    assert _cb_update_tune_visibility(False) == {"display": "none"}

    # Update Split Options
    assert _cb_update_split_options("group")[0] == {"display": "block"}
    assert _cb_update_split_options("timeseries")[1] == {"display": "block"}

    # Update Objectives
    assert len(_cb_update_tune_objectives("regression")) > 0
    assert len(_cb_update_tune_objectives("unknown")) == 0
