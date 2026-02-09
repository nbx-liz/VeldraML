import pytest
from pydantic import ValidationError

from veldra.config.models import RunConfig


def _base_payload() -> dict:
    return {
        "config_version": 1,
        "task": {"type": "frontier"},
        "data": {"target": "target"},
    }


def test_frontier_alpha_range_validation() -> None:
    payload = _base_payload()
    payload["frontier"] = {"alpha": 0.0}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)

    payload["frontier"] = {"alpha": 1.0}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)


def test_frontier_stratified_split_is_rejected() -> None:
    payload = _base_payload()
    payload["split"] = {"type": "stratified", "n_splits": 3, "seed": 42}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(payload)
