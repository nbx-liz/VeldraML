from __future__ import annotations

import pytest
from pydantic import ValidationError

from veldra.config.models import RunConfig


def _base_payload(task_type: str = "regression") -> dict:
    payload: dict = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": "train.csv", "target": "y"},
    }
    if task_type == "binary":
        payload["postprocess"] = {"calibration": "platt"}
    return payload


def test_num_boost_round_must_be_positive() -> None:
    payload = _base_payload()
    payload["train"] = {"num_boost_round": 0}
    with pytest.raises(ValidationError, match="train.num_boost_round must be >= 1"):
        RunConfig.model_validate(payload)


def test_early_stopping_validation_fraction_range() -> None:
    payload = _base_payload()
    payload["train"] = {"early_stopping_validation_fraction": 1.0}
    with pytest.raises(
        ValidationError, match="train.early_stopping_validation_fraction must satisfy 0 < value < 1"
    ):
        RunConfig.model_validate(payload)


def test_class_weight_is_classification_only() -> None:
    payload = _base_payload()
    payload["train"] = {"class_weight": {"0": 1.0, "1": 2.0}}
    with pytest.raises(ValidationError, match="train.class_weight can only be set"):
        RunConfig.model_validate(payload)


def test_auto_and_manual_class_weight_conflict_when_both_explicit() -> None:
    payload = _base_payload("binary")
    payload["train"] = {"auto_class_weight": True, "class_weight": {"0": 1.0, "1": 2.0}}
    with pytest.raises(ValidationError, match="cannot be enabled together"):
        RunConfig.model_validate(payload)


def test_binary_class_weight_value_must_be_positive() -> None:
    payload = _base_payload("binary")
    payload["train"] = {"class_weight": {"0": -1.0, "1": 2.0}}
    with pytest.raises(ValidationError, match="must be > 0"):
        RunConfig.model_validate(payload)


def test_auto_num_leaves_ratio_and_conflict_validation() -> None:
    payload = _base_payload()
    payload["train"] = {"auto_num_leaves": True, "num_leaves_ratio": 1.2}
    with pytest.raises(ValidationError, match="train.num_leaves_ratio must satisfy 0 < value <= 1"):
        RunConfig.model_validate(payload)

    payload = _base_payload()
    payload["train"] = {"auto_num_leaves": True, "lgb_params": {"num_leaves": 64}}
    with pytest.raises(
        ValidationError,
        match="cannot be combined with train.lgb_params.num_leaves",
    ):
        RunConfig.model_validate(payload)


def test_ratio_params_validation_and_conflicts() -> None:
    payload = _base_payload()
    payload["train"] = {"min_data_in_leaf_ratio": 0.0}
    with pytest.raises(
        ValidationError,
        match="train.min_data_in_leaf_ratio must satisfy 0 < value < 1",
    ):
        RunConfig.model_validate(payload)

    payload = _base_payload()
    payload["train"] = {"min_data_in_leaf_ratio": 0.2, "lgb_params": {"min_data_in_leaf": 10}}
    with pytest.raises(
        ValidationError,
        match="cannot be combined with train.lgb_params.min_data_in_leaf",
    ):
        RunConfig.model_validate(payload)

    payload = _base_payload()
    payload["train"] = {"min_data_in_bin_ratio": 0.2, "lgb_params": {"min_data_in_bin": 5}}
    with pytest.raises(
        ValidationError,
        match="cannot be combined with train.lgb_params.min_data_in_bin",
    ):
        RunConfig.model_validate(payload)


def test_feature_weights_value_validation() -> None:
    payload = _base_payload()
    payload["train"] = {"feature_weights": {"x1": 0.0}}
    with pytest.raises(ValidationError, match="train.feature_weights"):
        RunConfig.model_validate(payload)


def test_top_k_validation_and_scope() -> None:
    payload = _base_payload("regression")
    payload["train"] = {"top_k": 10}
    with pytest.raises(
        ValidationError,
        match="train.top_k can only be set when task.type='binary'",
    ):
        RunConfig.model_validate(payload)

    payload = _base_payload("binary")
    payload["train"] = {"top_k": 0}
    with pytest.raises(ValidationError, match="train.top_k must be >= 1"):
        RunConfig.model_validate(payload)


def test_precision_at_k_tuning_requires_top_k() -> None:
    payload = _base_payload("binary")
    payload["tuning"] = {"objective": "precision_at_k"}
    with pytest.raises(
        ValidationError,
        match="train.top_k is required when tuning.objective='precision_at_k'",
    ):
        RunConfig.model_validate(payload)
