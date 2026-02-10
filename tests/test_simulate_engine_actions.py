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


def test_normalize_scenarios_validation_errors() -> None:
    with pytest.raises(VeldraValidationError, match="dict or list"):
        normalize_scenarios("bad")
    with pytest.raises(VeldraValidationError, match="must not be empty"):
        normalize_scenarios([])
    with pytest.raises(VeldraValidationError, match="Each scenario must be a dict"):
        normalize_scenarios([1, 2])
    with pytest.raises(VeldraValidationError, match="non-empty string"):
        normalize_scenarios(
            [{"name": "   ", "actions": [{"op": "add", "column": "x1", "value": 1}]}]
        )
    with pytest.raises(VeldraValidationError, match="non-empty actions list"):
        normalize_scenarios([{"name": "s1", "actions": []}])


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


def test_apply_scenario_rejects_non_numeric_and_bad_values() -> None:
    frame = pd.DataFrame({"x1": [1.0], "x2": [2.0], "txt": ["a"], "target": [0.2]})
    with pytest.raises(VeldraValidationError, match="requires numeric column"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "add", "column": "txt", "value": 1}]},
            target_col="target",
            id_cols=[],
        )
    with pytest.raises(VeldraValidationError, match="non-empty column"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "add", "column": "", "value": 1}]},
            target_col="target",
            id_cols=[],
        )
    with pytest.raises(VeldraValidationError, match="set action requires numeric value"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "set", "column": "x1", "value": "z"}]},
            target_col="target",
            id_cols=[],
        )
    with pytest.raises(VeldraValidationError, match="add action requires numeric value"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "add", "column": "x1", "value": "z"}]},
            target_col="target",
            id_cols=[],
        )
    with pytest.raises(VeldraValidationError, match="mul action requires numeric value"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "mul", "column": "x1", "value": "z"}]},
            target_col="target",
            id_cols=[],
        )
    with pytest.raises(VeldraValidationError, match="requires min and/or max"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "clip", "column": "x1"}]},
            target_col="target",
            id_cols=[],
        )
    with pytest.raises(VeldraValidationError, match="clip min must be numeric or null"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "clip", "column": "x1", "min": "x"}]},
            target_col="target",
            id_cols=[],
        )
    with pytest.raises(VeldraValidationError, match="clip max must be numeric or null"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [{"op": "clip", "column": "x1", "max": "x"}]},
            target_col="target",
            id_cols=[],
        )


def test_apply_scenario_rejects_invalid_action_container() -> None:
    frame = pd.DataFrame({"x1": [1.0], "target": [0.2]})
    with pytest.raises(VeldraValidationError, match="invalid action type"):
        apply_scenario(
            frame,
            {"name": "bad", "actions": [1]},
            target_col="target",
            id_cols=[],
        )


def test_build_simulation_frame_binary_and_frontier_and_errors() -> None:
    row_ids = pd.Index([0, 1])
    binary_base = pd.DataFrame({"p_cal": [0.4, 0.8], "label_pred": [0, 1]})
    binary_scn = pd.DataFrame({"p_cal": [0.5, 0.6], "label_pred": [1, 1]})
    out_bin = build_simulation_frame(
        task_type="binary",
        row_ids=row_ids,
        scenario_name="s",
        base_pred=binary_base,
        scenario_pred=binary_scn,
    )
    assert {"base_p_cal", "scenario_p_cal", "delta_p_cal", "label_changed"} <= set(out_bin.columns)
    with pytest.raises(
        VeldraValidationError, match="Binary predict outputs must be pandas.DataFrame"
    ):
        build_simulation_frame(
            task_type="binary",
            row_ids=row_ids,
            scenario_name="s",
            base_pred=np.array([0.1, 0.2]),
            scenario_pred=binary_scn,
        )
    with pytest.raises(
        VeldraValidationError, match="Binary baseline output is missing required columns"
    ):
        build_simulation_frame(
            task_type="binary",
            row_ids=row_ids,
            scenario_name="s",
            base_pred=pd.DataFrame({"p_cal": [0.2, 0.3]}),
            scenario_pred=binary_scn,
        )

    frontier_base = pd.DataFrame({"frontier_pred": [1.0, 2.0], "u_hat": [0.1, 0.2]})
    frontier_scn = pd.DataFrame({"frontier_pred": [1.5, 1.8], "u_hat": [0.4, 0.1]})
    out_frontier = build_simulation_frame(
        task_type="frontier",
        row_ids=row_ids,
        scenario_name="s",
        base_pred=frontier_base,
        scenario_pred=frontier_scn,
    )
    assert {"base_pred", "scenario_pred", "delta_pred", "base_u_hat", "scenario_u_hat"} <= set(
        out_frontier.columns
    )
    with pytest.raises(
        VeldraValidationError, match="Frontier predict outputs must be pandas.DataFrame"
    ):
        build_simulation_frame(
            task_type="frontier",
            row_ids=row_ids,
            scenario_name="s",
            base_pred=np.array([1.0, 2.0]),
            scenario_pred=frontier_scn,
        )
    with pytest.raises(VeldraValidationError, match="Unsupported task type"):
        build_simulation_frame(
            task_type="unknown",
            row_ids=row_ids,
            scenario_name="s",
            base_pred=np.array([1.0, 2.0]),
            scenario_pred=np.array([1.1, 2.1]),
        )


def test_build_simulation_frame_multiclass_validation_errors() -> None:
    row_ids = pd.Index([0, 1])
    base = pd.DataFrame({"label_pred": ["a", "b"], "proba_a": [0.6, 0.1], "proba_b": [0.4, 0.9]})
    scn = pd.DataFrame({"label_pred": ["a", "a"], "proba_a": [0.7, 0.2], "proba_b": [0.3, 0.8]})
    with pytest.raises(
        VeldraValidationError, match="Multiclass simulation requires target classes"
    ):
        build_simulation_frame(
            task_type="multiclass",
            row_ids=row_ids,
            scenario_name="s",
            base_pred=pd.DataFrame({"label_pred": ["a", "b"]}),
            scenario_pred=pd.DataFrame({"label_pred": ["a", "a"]}),
            target_classes=[],
        )
    with pytest.raises(
        VeldraValidationError, match="Multiclass scenario output is missing required columns"
    ):
        build_simulation_frame(
            task_type="multiclass",
            row_ids=row_ids,
            scenario_name="s",
            base_pred=base,
            scenario_pred=pd.DataFrame({"label_pred": ["a", "a"], "proba_a": [0.5, 0.5]}),
            target_classes=["a", "b"],
        )
    with pytest.raises(
        VeldraValidationError, match="Multiclass predict outputs must be pandas.DataFrame"
    ):
        build_simulation_frame(
            task_type="multiclass",
            row_ids=row_ids,
            scenario_name="s",
            base_pred=np.array([0.1, 0.2]),
            scenario_pred=scn,
            target_classes=["a", "b"],
        )
