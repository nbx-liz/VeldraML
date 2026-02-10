from __future__ import annotations

import pytest

from veldra.config.models import RunConfig
from veldra.modeling import tuning


def _frontier_cfg(
    *,
    objective: str,
    coverage_target: float | None = None,
    coverage_tolerance: float = 0.01,
    penalty_weight: float = 1.0,
) -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "frontier": {"alpha": 0.90},
            "tuning": {
                "enabled": True,
                "n_trials": 1,
                "objective": objective,
                "coverage_target": coverage_target,
                "coverage_tolerance": coverage_tolerance,
                "penalty_weight": penalty_weight,
            },
        }
    )


def test_frontier_penalty_is_zero_within_tolerance() -> None:
    cfg = _frontier_cfg(
        objective="pinball_coverage_penalty",
        coverage_target=0.90,
        coverage_tolerance=0.02,
        penalty_weight=3.0,
    )
    value, components = tuning._frontier_objective_from_metrics(
        cfg,
        "pinball_coverage_penalty",
        {"pinball": 0.25, "coverage": 0.91},
    )
    assert value == pytest.approx(0.25)
    assert components["penalty"] == 0.0
    assert components["objective_value"] == 0.25


def test_frontier_penalty_adds_beyond_tolerance() -> None:
    cfg = _frontier_cfg(
        objective="pinball_coverage_penalty",
        coverage_target=0.90,
        coverage_tolerance=0.01,
        penalty_weight=2.0,
    )
    value, components = tuning._frontier_objective_from_metrics(
        cfg,
        "pinball_coverage_penalty",
        {"pinball": 0.20, "coverage": 0.95},
    )
    # gap 0.05, tolerance 0.01 => penalized gap 0.04, weight 2 => penalty 0.08
    assert value == pytest.approx(0.28)
    assert components["penalty"] == pytest.approx(0.08)
    assert components["coverage_gap"] == pytest.approx(0.05)


def test_frontier_pinball_objective_ignores_penalty_but_keeps_components() -> None:
    cfg = _frontier_cfg(
        objective="pinball",
        coverage_target=0.85,
        coverage_tolerance=0.0,
        penalty_weight=10.0,
    )
    value, components = tuning._frontier_objective_from_metrics(
        cfg,
        "pinball",
        {"pinball": 0.15, "coverage": 0.95},
    )
    assert value == pytest.approx(0.15)
    assert components["pinball"] == pytest.approx(0.15)
    assert components["objective_value"] == pytest.approx(0.15)
