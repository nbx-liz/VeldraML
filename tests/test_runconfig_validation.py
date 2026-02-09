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
