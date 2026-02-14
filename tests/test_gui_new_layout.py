import base64
from unittest.mock import MagicMock, patch

import plotly.graph_objs as go
import yaml

from veldra.gui.app import (
    _cb_build_config_yaml,
    _cb_detect_run_action,
    _cb_inspect_data,
    _cb_populate_builder_options,
    _cb_update_result_view,
)

# Mock dash callback_context if needed, though for these unittests maybe not.

# --- Test Data Page (Upload) ---


def test_inspect_data_upload_csv():
    """Test inspecting uploaded CSV content."""
    # Create dummy CSV content
    csv_content = "col1,col2,target\n1,2,0\n3,4,1"
    encoded = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    content_string = f"data:text/csv;base64,{encoded}"

    with patch("veldra.gui.app.inspect_data") as mock_inspect:
        # Mock inspect_data to return success
        mock_inspect.return_value = {
            "success": True,
            "stats": {
                "n_rows": 2,
                "n_cols": 3,
                "columns": ["col1", "col2", "target"],
                "numeric_cols": [],
                "categorical_cols": [],
                "missing_count": 0,
            },
            "preview": [{"col1": 1, "col2": 2, "target": 0}],
        }
        # Mock rendering functions inside data_page
        with patch("veldra.gui.pages.data_page.render_data_stats"):
            with patch("veldra.gui.pages.data_page.render_data_preview"):
                # Call callback
                # args: n_clicks, file_path, upload_contents, upload_filename
                out_div, msg, state = _cb_inspect_data(1, "", content_string, "test_upload.csv")

                # Check that final data path is temp path
                assert "temp_data" in state["data_path"]
                assert "test_upload.csv" in state["data_path"]
                assert mock_inspect.called


def test_inspect_data_upload_invalid():
    """Test uploading unsupported file type."""
    content_string = "data:application/json;base64,ew==..."
    out, msg, state = _cb_inspect_data(1, "", content_string, "test.json")
    assert "Unsupported file type" in msg


# --- Test Config Builder (New Fields) ---


def test_build_config_yaml_causal_target_search_space():
    """Test generating YAML with new Causal, Target, and Search Space parameters."""
    # The signature of _cb_build_config_yaml is huge. We need to match it.
    # task_type, d_path, d_target, d_ids, d_cats, d_drops,
    # s_type, s_n, s_seed, s_grp, s_time, s_ts_mode, s_test, s_gap, s_embargo,
    # t_lr, t_leaves, t_est, t_depth, t_child, t_early, t_sub, t_col, t_l1, t_l2,
    # tune_en, tune_pre, tune_tri, tune_obj,
    # exp_dir,
    # causal_en, causal_method,
    # tune_lr_min, tune_lr_max, tune_leaves_min, tune_leaves_max,
    # tune_depth_min, tune_depth_max, tune_ff_min, tune_ff_max

    yaml_str = _cb_build_config_yaml(
        "regression",
        "data/path.csv",
        "target_col",
        [],
        [],
        [],  # task, data...
        "kfold",
        5,
        42,
        None,
        None,
        "expanding",
        None,
        0,
        0,  # split...
        0.1,
        31,
        100,
        -1,
        20,
        100,
        1.0,
        1.0,
        0,
        0,  # train...
        True,
        "standard",
        30,
        "rmse",  # tune
        "artifacts",  # export
        True,
        "dr",  # causal
        0.01,
        0.2,
        16,
        64,
        None,
        None,
        None,
        None,  # search space
    )

    cfg = yaml.safe_load(yaml_str)

    # Verify Causal
    assert cfg["task"]["type"] == "regression"
    assert cfg["task"]["causal_method"] == "dr"

    # Verify Target (moved to config)
    assert cfg["data"]["target"] == "target_col"

    # Verify Search Space
    assert cfg["tuning"]["enabled"] is True
    assert "search_space" in cfg["tuning"]
    space = cfg["tuning"]["search_space"]
    # Check learning_rate range
    assert space["learning_rate"]["low"] == 0.01
    assert space["learning_rate"]["high"] == 0.2
    assert space["learning_rate"]["log"] is True
    # Check leaves range
    assert space["num_leaves"]["low"] == 16
    assert space["num_leaves"]["high"] == 64
    # Check unused params are not in dict
    assert "max_depth" not in space  # None values


# --- Test Run Auto-Action ---


def test_detect_run_action_tune():
    """Test detecting TUNE action."""
    cfg = {"tuning": {"enabled": True}}
    yaml_text = yaml.dump(cfg)

    # args: yaml_text
    # returns: action, text, className
    action, text, style = _cb_detect_run_action(yaml_text)

    assert action == "tune"
    assert "TUNE" in text
    assert "warning" in style


def test_detect_run_action_causal():
    """Test detecting CAUSAL FIT action."""
    cfg = {"task": {"type": "regression", "causal_method": "dr"}, "tuning": {"enabled": False}}
    yaml_text = yaml.dump(cfg)

    action, text, style = _cb_detect_run_action(yaml_text)

    assert action == "fit"  # Action is fit, but label changes
    assert "CAUSAL" in text
    assert "info" in style


def test_detect_run_action_default():
    """Test default FIT action."""
    cfg = {"task": {"type": "regression"}, "tuning": {"enabled": False}}
    yaml_text = yaml.dump(cfg)
    action, text, style = _cb_detect_run_action(yaml_text)
    assert action == "fit"
    assert "TRAIN" in text
    assert "primary" in style


# --- Test Result Page Fix ---


def test_update_result_view_chart_fix():
    """Test that missing feature importance returns valid empty figure, not dict."""
    with patch("veldra.gui.app.Artifact.load") as mock_load:
        # Mock Artifact
        mock_art = MagicMock()
        mock_art.metadata = {}  # No feature importance
        mock_art.metrics = {"rmse": 0.5}
        mock_art.config = {}
        mock_load.return_value = mock_art

        # args: artifact_path, compare_path
        kpi, fig_main, fig_sec, details = _cb_update_result_view("path/to/art", None)

        # fig_sec should be a go.Figure object
        # It shouldn't be an empty dict {}
        assert isinstance(fig_sec, go.Figure) or (isinstance(fig_sec, dict) and "data" in fig_sec)
        assert fig_sec != {}


# --- Test Builder Options Populate ---


def test_populate_builder_options():
    """Test populating builder options from state."""
    state = {"data_path": "test.csv", "target_col": "target"}
    with patch("veldra.gui.app.inspect_data") as mock_inspect:
        mock_inspect.return_value = {
            "success": True,
            "stats": {"columns": ["col1", "col2", "target"]},
        }

        path, target, opts_id, opts_cat, opts_drop, opts_grp, opts_time = (
            _cb_populate_builder_options("/", state)
        )

        assert path == "test.csv"
        assert target == "target"  # Should retrieve saved target
        # Check non-target options
        assert len(opts_drop) == 2  # col1, col2
        assert {"label": "target", "value": "target"} not in opts_drop
