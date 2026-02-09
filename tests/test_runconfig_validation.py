import pytest
from pydantic import ValidationError

from veldra.config.models import RunConfig


def _minimal_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"target": "y"},
    }


def test_runconfig_requires_config_version() -> None:
    payload = _minimal_payload()
    payload.pop("config_version")

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_postprocess_binary_only() -> None:
    payload = _minimal_payload()
    payload["postprocess"] = {"calibration": "platt"}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_timeseries_requires_time_col() -> None:
    payload = _minimal_payload()
    payload["split"] = {"type": "timeseries", "n_splits": 3}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_group_split_requires_group_col() -> None:
    payload = _minimal_payload()
    payload["split"] = {"type": "group", "n_splits": 3}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_binary_calibration_allows_only_platt_in_current_phase() -> None:
    payload = _minimal_payload()
    payload["task"] = {"type": "binary"}
    payload["postprocess"] = {"calibration": "isotonic"}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_binary_threshold_must_be_between_zero_and_one() -> None:
    payload = _minimal_payload()
    payload["task"] = {"type": "binary"}
    payload["postprocess"] = {"threshold": 1.5}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_multiclass_disallows_postprocess_calibration_and_threshold() -> None:
    payload = _minimal_payload()
    payload["task"] = {"type": "multiclass"}
    payload["postprocess"] = {"calibration": "platt", "threshold": 0.5}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_threshold_optimization_is_binary_only() -> None:
    payload = _minimal_payload()
    payload["postprocess"] = {"threshold_optimization": {"enabled": True, "objective": "f1"}}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_threshold_optimization_conflicts_with_fixed_threshold() -> None:
    payload = _minimal_payload()
    payload["task"] = {"type": "binary"}
    payload["postprocess"] = {
        "threshold": 0.5,
        "threshold_optimization": {"enabled": True, "objective": "f1"},
    }

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_tuning_objective_is_task_constrained() -> None:
    payload = _minimal_payload()
    payload["tuning"] = {"enabled": True, "n_trials": 1, "objective": "rmse"}
    RunConfig.model_validate(payload)

    payload["tuning"]["objective"] = "auc"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_frontier_alpha_must_be_in_open_interval() -> None:
    payload = _minimal_payload()
    payload["task"] = {"type": "frontier"}
    payload["frontier"] = {"alpha": 1.0}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_frontier_disallows_stratified_split() -> None:
    payload = _minimal_payload()
    payload["task"] = {"type": "frontier"}
    payload["split"] = {"type": "stratified", "n_splits": 3, "seed": 42}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_non_frontier_disallows_custom_frontier_settings() -> None:
    payload = _minimal_payload()
    payload["frontier"] = {"alpha": 0.8}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)
