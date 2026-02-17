from __future__ import annotations

from veldra.modeling.tuning import _default_search_space


def test_standard_search_space_matches_phase263_contract() -> None:
    search_space = _default_search_space("regression", "standard")

    assert search_space["learning_rate"] == {
        "type": "float",
        "low": 0.01,
        "high": 0.1,
        "log": True,
    }
    assert search_space["train.num_leaves_ratio"] == {
        "type": "float",
        "low": 0.5,
        "high": 1.0,
    }
    assert search_space["train.early_stopping_validation_fraction"] == {
        "type": "float",
        "low": 0.1,
        "high": 0.3,
    }
    assert search_space["max_bin"] == {
        "type": "int",
        "low": 127,
        "high": 255,
    }
    assert search_space["train.min_data_in_leaf_ratio"] == {
        "type": "float",
        "low": 0.01,
        "high": 0.1,
    }
    assert search_space["train.min_data_in_bin_ratio"] == {
        "type": "float",
        "low": 0.01,
        "high": 0.1,
    }
    assert search_space["max_depth"] == {"type": "int", "low": 3, "high": 15}
    assert search_space["feature_fraction"] == {"type": "float", "low": 0.5, "high": 1.0}
    assert search_space["bagging_fraction"] == 1.0
    assert search_space["bagging_freq"] == 0
    assert search_space["lambda_l1"] == 0.0
    assert search_space["lambda_l2"] == {"type": "float", "low": 0.000001, "high": 0.1}
