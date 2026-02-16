from __future__ import annotations

import pandas as pd

from veldra.gui import app as app_module


def test_e2e_state_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "_get_load_tabular_data",
        lambda: (
            lambda _p: pd.DataFrame(
                {
                    "x1": [1, 2, 3, 4],
                    "x2": [4, 3, 2, 1],
                    "target": [0, 1, 0, 1],
                }
            )
        ),
    )

    state = app_module._ensure_workflow_state_defaults({"data_path": "train.csv"})

    state = app_module._cb_save_target_state(
        "target",
        "binary",
        ["x2"],
        False,
        None,
        None,
        None,
        state,
    )
    state = app_module._cb_save_validation_state(
        "stratified",
        5,
        42,
        None,
        None,
        "expanding",
        None,
        0,
        0,
        state,
    )
    state = app_module._cb_save_train_state(
        0.05,
        300,
        31,
        -1,
        20,
        100,
        True,
        "",
        False,
        "standard",
        30,
        None,
        "artifacts",
        state,
    )

    yaml_text = state["config_yaml"]
    assert "task:" in yaml_text
    assert "binary" in yaml_text
    assert "drop_cols" in yaml_text
