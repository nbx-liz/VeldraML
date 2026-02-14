import pandas as pd
import pytest

from veldra.api import Artifact, evaluate, fit
from veldra.api.exceptions import VeldraValidationError


def test_frontier_evaluate_returns_expected_metrics(tmp_path, frontier_frame) -> None:
    frame = frontier_frame(rows=110, seed=61)
    data_path = tmp_path / "frontier.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 7},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)

    result = evaluate(artifact, frame)
    assert {"pinball", "mae", "mean_u_hat", "coverage"} <= set(result.metrics.keys())
    assert result.metadata["frontier_alpha"] == pytest.approx(0.90)


def test_frontier_evaluate_rejects_invalid_input(tmp_path, frontier_frame) -> None:
    frame = frontier_frame(rows=110, seed=61)
    data_path = tmp_path / "frontier.csv"
    frame.to_csv(data_path, index=False)
    run = fit(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 7},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)

    with pytest.raises(VeldraValidationError):
        evaluate(artifact, frame.drop(columns=["target"]))
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, pd.DataFrame())
    with pytest.raises(VeldraValidationError):
        evaluate(artifact, data=[1, 2, 3])  # type: ignore[arg-type]
