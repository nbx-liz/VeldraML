from __future__ import annotations

from veldra.modeling.tuning import _default_search_space


def test_standard_search_space_includes_phase258_parameters() -> None:
    search_space = _default_search_space("regression", "standard")

    assert search_space["lambda_l1"] == {
        "type": "float",
        "low": 1e-8,
        "high": 10.0,
        "log": True,
    }
    assert search_space["lambda_l2"] == {
        "type": "float",
        "low": 1e-8,
        "high": 10.0,
        "log": True,
    }
    assert search_space["path_smooth"] == {
        "type": "float",
        "low": 0.0,
        "high": 10.0,
    }
    assert search_space["min_gain_to_split"] == {
        "type": "float",
        "low": 0.0,
        "high": 1.0,
    }
