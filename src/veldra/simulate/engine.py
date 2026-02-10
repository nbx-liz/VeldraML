"""Scenario simulation engine utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from veldra.api.exceptions import VeldraValidationError

_ALLOWED_OPS = {"set", "add", "mul", "clip"}


def normalize_scenarios(scenarios: Any) -> list[dict[str, Any]]:
    """Normalize incoming scenarios into a non-empty list."""
    if isinstance(scenarios, dict):
        items = [dict(scenarios)]
    elif isinstance(scenarios, list):
        items = [dict(item) if isinstance(item, dict) else item for item in scenarios]
    else:
        raise VeldraValidationError("scenarios must be a dict or list[dict].")

    if not items:
        raise VeldraValidationError("scenarios must not be empty.")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise VeldraValidationError("Each scenario must be a dict.")
        name = item.get("name")
        if name is None:
            name = f"scenario_{idx}"
        if not isinstance(name, str) or not name.strip():
            raise VeldraValidationError("scenario.name must be a non-empty string.")
        actions = item.get("actions")
        if not isinstance(actions, list) or not actions:
            raise VeldraValidationError(
                f"Scenario '{name}' must include a non-empty actions list."
            )
        normalized.append({"name": name, "actions": actions})
    return normalized


def _validate_numeric_column(df: pd.DataFrame, column: str, scenario_name: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise VeldraValidationError(
            f"Scenario '{scenario_name}' requires numeric column '{column}'."
        )


def apply_scenario(
    data: pd.DataFrame,
    scenario: dict[str, Any],
    *,
    target_col: str,
    id_cols: list[str],
) -> pd.DataFrame:
    """Apply scenario actions to a copied DataFrame."""
    scenario_name = str(scenario["name"])
    actions = scenario["actions"]
    out = data.copy()
    forbidden_cols = {target_col, *id_cols}

    for action in actions:
        if not isinstance(action, dict):
            raise VeldraValidationError(
                f"Scenario '{scenario_name}' includes an invalid action type."
            )
        op = action.get("op")
        column = action.get("column")
        if op not in _ALLOWED_OPS:
            raise VeldraValidationError(
                f"Scenario '{scenario_name}' has unsupported op '{op}'. "
                f"Allowed: {sorted(_ALLOWED_OPS)}"
            )
        if not isinstance(column, str) or not column:
            raise VeldraValidationError(
                f"Scenario '{scenario_name}' action must include non-empty column."
            )
        if column not in out.columns:
            raise VeldraValidationError(
                f"Scenario '{scenario_name}' references unknown column '{column}'."
            )
        if column in forbidden_cols:
            raise VeldraValidationError(
                f"Scenario '{scenario_name}' cannot modify protected column '{column}'."
            )
        _validate_numeric_column(out, column, scenario_name)

        if op == "set":
            value = action.get("value")
            if not isinstance(value, (int, float)):
                raise VeldraValidationError(
                    f"Scenario '{scenario_name}' set action requires numeric value."
                )
            out[column] = float(value)
        elif op == "add":
            value = action.get("value")
            if not isinstance(value, (int, float)):
                raise VeldraValidationError(
                    f"Scenario '{scenario_name}' add action requires numeric value."
                )
            out[column] = out[column] + float(value)
        elif op == "mul":
            value = action.get("value")
            if not isinstance(value, (int, float)):
                raise VeldraValidationError(
                    f"Scenario '{scenario_name}' mul action requires numeric value."
                )
            out[column] = out[column] * float(value)
        else:
            min_v = action.get("min")
            max_v = action.get("max")
            if min_v is None and max_v is None:
                raise VeldraValidationError(
                    f"Scenario '{scenario_name}' clip action requires min and/or max."
                )
            if min_v is not None and not isinstance(min_v, (int, float)):
                raise VeldraValidationError(
                    f"Scenario '{scenario_name}' clip min must be numeric or null."
                )
            if max_v is not None and not isinstance(max_v, (int, float)):
                raise VeldraValidationError(
                    f"Scenario '{scenario_name}' clip max must be numeric or null."
                )
            out[column] = out[column].clip(lower=min_v, upper=max_v)

    return out


def _as_vector(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise VeldraValidationError(f"{label} output must be 1-dimensional.")
    return arr


def _ensure_columns(
    frame: pd.DataFrame,
    required: list[str],
    *,
    label: str,
) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise VeldraValidationError(f"{label} output is missing required columns: {missing}")


def build_simulation_frame(
    *,
    task_type: str,
    row_ids: pd.Index,
    scenario_name: str,
    base_pred: Any,
    scenario_pred: Any,
    target_classes: list[Any] | None = None,
) -> pd.DataFrame:
    """Build task-specific simulation comparison frame."""
    result = pd.DataFrame(
        {
            "row_id": row_ids.to_list(),
            "scenario": scenario_name,
            "task_type": task_type,
        }
    )
    if task_type == "regression":
        base_vec = _as_vector(base_pred, "Regression baseline")
        scenario_vec = _as_vector(scenario_pred, "Regression scenario")
        result["base_pred"] = base_vec
        result["scenario_pred"] = scenario_vec
        result["delta_pred"] = scenario_vec - base_vec
        return result

    if task_type == "binary":
        if not isinstance(base_pred, pd.DataFrame) or not isinstance(scenario_pred, pd.DataFrame):
            raise VeldraValidationError("Binary predict outputs must be pandas.DataFrame.")
        required = ["p_cal", "label_pred"]
        _ensure_columns(base_pred, required, label="Binary baseline")
        _ensure_columns(scenario_pred, required, label="Binary scenario")
        result["base_p_cal"] = base_pred["p_cal"].to_numpy(dtype=float)
        result["scenario_p_cal"] = scenario_pred["p_cal"].to_numpy(dtype=float)
        result["delta_p_cal"] = result["scenario_p_cal"] - result["base_p_cal"]
        base_label = base_pred["label_pred"].to_numpy(dtype=int)
        scenario_label = scenario_pred["label_pred"].to_numpy(dtype=int)
        result["base_label_pred"] = base_label
        result["scenario_label_pred"] = scenario_label
        result["label_changed"] = base_label != scenario_label
        return result

    if task_type == "multiclass":
        if not isinstance(base_pred, pd.DataFrame) or not isinstance(scenario_pred, pd.DataFrame):
            raise VeldraValidationError("Multiclass predict outputs must be pandas.DataFrame.")
        if not target_classes:
            target_classes = [
                col.replace("proba_", "")
                for col in base_pred.columns
                if isinstance(col, str) and col.startswith("proba_")
            ]
        if not target_classes:
            raise VeldraValidationError("Multiclass simulation requires target classes.")
        prob_cols = [f"proba_{label}" for label in target_classes]
        _ensure_columns(base_pred, ["label_pred", *prob_cols], label="Multiclass baseline")
        _ensure_columns(scenario_pred, ["label_pred", *prob_cols], label="Multiclass scenario")
        base_label = base_pred["label_pred"].to_numpy()
        scenario_label = scenario_pred["label_pred"].to_numpy()
        result["base_label_pred"] = base_label
        result["scenario_label_pred"] = scenario_label
        result["label_changed"] = base_label != scenario_label
        for col in prob_cols:
            result[f"base_{col}"] = base_pred[col].to_numpy(dtype=float)
            result[f"scenario_{col}"] = scenario_pred[col].to_numpy(dtype=float)
            result[f"delta_{col}"] = result[f"scenario_{col}"] - result[f"base_{col}"]
        return result

    if task_type == "frontier":
        if not isinstance(base_pred, pd.DataFrame) or not isinstance(scenario_pred, pd.DataFrame):
            raise VeldraValidationError("Frontier predict outputs must be pandas.DataFrame.")
        _ensure_columns(base_pred, ["frontier_pred"], label="Frontier baseline")
        _ensure_columns(scenario_pred, ["frontier_pred"], label="Frontier scenario")
        result["base_pred"] = base_pred["frontier_pred"].to_numpy(dtype=float)
        result["scenario_pred"] = scenario_pred["frontier_pred"].to_numpy(dtype=float)
        result["delta_pred"] = result["scenario_pred"] - result["base_pred"]
        if "u_hat" in base_pred.columns and "u_hat" in scenario_pred.columns:
            result["base_u_hat"] = base_pred["u_hat"].to_numpy(dtype=float)
            result["scenario_u_hat"] = scenario_pred["u_hat"].to_numpy(dtype=float)
            result["delta_u_hat"] = result["scenario_u_hat"] - result["base_u_hat"]
        return result

    raise VeldraValidationError(f"Unsupported task type for simulation: '{task_type}'")
