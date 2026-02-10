from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.simulate.engine import apply_scenario, build_simulation_frame, normalize_scenarios


def test_apply_scenario_supports_set_add_mul_clip() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x1": [1.0, 2.0, 3.0],
            "x2": [10.0, 20.0, 30.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    scenario = {
        "name": "ops",
        "actions": [
            {"op": "set", "column": "x1", "value": 4.0},
            {"op": "add", "column": "x1", "value": 2.0},
            {"op": "mul", "column": "x2", "value": 0.5},
            {"op": "clip", "column": "x2", "min": 8.0, "max": 12.0},
        ],
    }

    out = apply_scenario(frame, scenario, target_col="target", id_cols=["id"])
    assert out["x1"].tolist() == [6.0, 6.0, 6.0]
    assert out["x2"].tolist() == [8.0, 10.0, 12.0]
    assert out["target"].tolist() == frame["target"].tolist()


def test_apply_scenario_rejects_protected_or_invalid_actions() -> None:
    frame = pd.DataFrame({"id": [1], "x1": [1.0], "target": [0.2]})

    with pytest.raises(VeldraValidationError, match="unsupported op"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "pow", "column": "x1", "value": 2}]},
            target_col="target",
            id_cols=["id"],
        )
    with pytest.raises(VeldraValidationError, match="unknown column"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "add", "column": "x9", "value": 1}]},
            target_col="target",
            id_cols=["id"],
        )
    with pytest.raises(VeldraValidationError, match="protected column 'target'"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "set", "column": "target", "value": 0.5}]},
            target_col="target",
            id_cols=["id"],
        )
    with pytest.raises(VeldraValidationError, match="protected column 'id'"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "set", "column": "id", "value": 2}]},
            target_col="target",
            id_cols=["id"],
        )


def test_normalize_scenarios_accepts_dict_and_generates_name() -> None:
    scenarios = normalize_scenarios({"actions": [{"op": "add", "column": "x1", "value": 1.0}]})
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "scenario_1"


def test_build_simulation_frame_multiclass_contract() -> None:
    base = pd.DataFrame(
        {
            "label_pred": ["a", "b"],
            "proba_a": [0.7, 0.1],
            "proba_b": [0.2, 0.8],
            "proba_c": [0.1, 0.1],
        }
    )
    scenario = pd.DataFrame(
        {
            "label_pred": ["a", "c"],
            "proba_a": [0.6, 0.1],
            "proba_b": [0.2, 0.2],
            "proba_c": [0.2, 0.7],
        }
    )
    out = build_simulation_frame(
        task_type="multiclass",
        row_ids=pd.Index([10, 11]),
        scenario_name="case_a",
        base_pred=base,
        scenario_pred=scenario,
        target_classes=["a", "b", "c"],
    )
    assert {"base_label_pred", "scenario_label_pred", "label_changed"} <= set(out.columns)
    assert {"base_proba_a", "scenario_proba_a", "delta_proba_a"} <= set(out.columns)
    assert out["label_changed"].tolist() == [False, True]


def test_build_simulation_frame_rejects_wrong_prediction_shape() -> None:
    with pytest.raises(VeldraValidationError, match="1-dimensional"):
        build_simulation_frame(
            task_type="regression",
            row_ids=pd.Index([0, 1]),
            scenario_name="bad",
            base_pred=np.array([[1.0, 2.0]]),
            scenario_pred=np.array([1.2, 2.2]),
        )
