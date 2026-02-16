from __future__ import annotations

from veldra.gui import app as app_module


def test_workflow_defaults_and_config_build() -> None:
    state = app_module._ensure_workflow_state_defaults({"data_path": "train.csv"})
    assert state["task_type"] == "regression"
    assert "split_config" in state

    state["target_col"] = "target"
    yaml_text = app_module._build_config_from_state(state)
    assert "config_version: 1" in yaml_text
    assert "target: target" in yaml_text


def test_state_from_config_payload_roundtrip() -> None:
    payload = {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": "x.csv", "target": "target", "drop_cols": ["id"]},
        "split": {"type": "stratified", "n_splits": 3, "seed": 7},
        "train": {
            "num_boost_round": 120,
            "lgb_params": {
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
            },
            "early_stopping_rounds": 30,
            "auto_class_weight": True,
        },
        "tuning": {"enabled": False},
        "export": {"artifact_dir": "artifacts"},
    }
    state = app_module._state_from_config_payload(payload)
    assert state["task_type"] == "binary"
    assert state["target_col"] == "target"
    assert state["split_config"]["type"] == "stratified"
