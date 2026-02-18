from __future__ import annotations

import math

import pytest

from veldra.api import Artifact, fit
from veldra.diagnostics.metrics import frontier_metrics


@pytest.mark.parametrize("alpha", [0.01, 0.99])
def test_frontier_fit_handles_alpha_boundaries(tmp_path, frontier_frame, alpha: float) -> None:
    frame = frontier_frame(rows=100, seed=801)
    path = tmp_path / f"frontier_alpha_{alpha}.csv"
    frame.to_csv(path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 24},
            "frontier": {"alpha": alpha},
            "train": {"num_boost_round": 45, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert 0.0 <= float(run.metrics["coverage"]) <= 1.0
    assert math.isfinite(float(run.metrics["pinball"]))

    artifact = Artifact.load(run.artifact_path)
    assert artifact.observation_table is not None
    assert "efficiency" in artifact.observation_table.columns


def test_frontier_metrics_single_observation_contract() -> None:
    metrics = frontier_metrics([1.0], [1.2], alpha=0.9, label="single")
    assert metrics["label"] == "single"
    assert math.isfinite(float(metrics["pinball"]))
    assert 0.0 <= float(metrics["coverage"]) <= 1.0
