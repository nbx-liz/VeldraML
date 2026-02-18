from __future__ import annotations

import yaml

from veldra.gui import app as app_module


def test_config_payload_from_state_contains_sections() -> None:
    state = {
        "data_path": "train.csv",
        "target_col": "target",
        "task_type": "binary",
        "exclude_cols": ["id"],
        "split_config": {"type": "stratified", "n_splits": 5, "seed": 42},
        "train_config": {
            "learning_rate": 0.05,
            "num_boost_round": 300,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "early_stopping_rounds": 100,
            "auto_class_weight": True,
            "class_weight_text": "",
        },
        "tuning_config": {"enabled": True, "preset": "fast", "n_trials": 10, "objective": "auc"},
        "artifact_dir": "artifacts",
    }

    payload = app_module._build_config_payload_from_state(state)
    assert payload["task"]["type"] == "binary"
    assert payload["split"]["type"] == "stratified"
    assert payload["tuning"]["enabled"] is True

    yaml_text = app_module._build_config_from_state(state)
    loaded = yaml.safe_load(yaml_text)
    assert loaded["data"]["target"] == "target"
    assert loaded["export"]["artifact_dir"] == "artifacts"
