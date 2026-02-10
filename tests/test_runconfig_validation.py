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


def test_timeseries_blocked_requires_train_size() -> None:
    payload = _minimal_payload()
    payload["split"] = {
        "type": "timeseries",
        "n_splits": 3,
        "time_col": "ts",
        "timeseries_mode": "blocked",
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_timeseries_expanding_disallows_train_size() -> None:
    payload = _minimal_payload()
    payload["split"] = {
        "type": "timeseries",
        "n_splits": 3,
        "time_col": "ts",
        "timeseries_mode": "expanding",
        "train_size": 10,
    }
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_non_timeseries_disallows_timeseries_specific_split_params() -> None:
    payload = _minimal_payload()
    payload["split"] = {"type": "kfold", "n_splits": 3, "gap": 1}
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

    payload["task"] = {"type": "frontier"}
    payload["split"] = {"type": "kfold", "n_splits": 2, "seed": 42}
    payload["frontier"] = {"alpha": 0.90}
    payload["tuning"]["objective"] = "pinball"
    RunConfig.model_validate(payload)

    payload["tuning"]["objective"] = "pinball_coverage_penalty"
    RunConfig.model_validate(payload)

    payload["tuning"]["objective"] = "rmse"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_frontier_tuning_coverage_fields_validation() -> None:
    payload = _minimal_payload()
    payload["task"] = {"type": "frontier"}
    payload["split"] = {"type": "kfold", "n_splits": 2, "seed": 42}
    payload["frontier"] = {"alpha": 0.90}
    payload["tuning"] = {
        "enabled": True,
        "n_trials": 1,
        "objective": "pinball_coverage_penalty",
        "coverage_target": 0.92,
        "coverage_tolerance": 0.02,
        "penalty_weight": 2.0,
    }
    RunConfig.model_validate(payload)

    payload["tuning"]["coverage_target"] = 1.2
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)

    payload["tuning"]["coverage_target"] = 0.92
    payload["tuning"]["coverage_tolerance"] = -0.1
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)

    payload["tuning"]["coverage_tolerance"] = 0.02
    payload["tuning"]["penalty_weight"] = -1.0
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_non_frontier_disallows_frontier_tuning_coverage_fields() -> None:
    payload = _minimal_payload()
    payload["tuning"] = {"enabled": True, "n_trials": 1, "coverage_target": 0.9}
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


def test_export_onnx_optimization_defaults_to_disabled() -> None:
    cfg = RunConfig.model_validate(_minimal_payload())
    assert cfg.export.onnx_optimization.enabled is False
    assert cfg.export.onnx_optimization.mode == "dynamic_quant"


def test_export_onnx_optimization_requires_mode_when_enabled() -> None:
    payload = _minimal_payload()
    payload["export"] = {"onnx_optimization": {"enabled": True, "mode": None}}

    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)
