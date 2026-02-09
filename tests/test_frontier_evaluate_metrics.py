import numpy as np
import pandas as pd
import pytest

from veldra.api import Artifact, evaluate, fit
from veldra.api.exceptions import VeldraValidationError


def _frontier_frame(rows: int = 110, seed: int = 61) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.8, 2.3, size=rows)
    x2 = rng.normal(size=rows)
    y = 1.7 + 0.9 * x1 - 0.6 * x2 + rng.normal(scale=0.3, size=rows) + rng.exponential(
        scale=0.25, size=rows
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def test_frontier_evaluate_returns_expected_metrics(tmp_path) -> None:
    frame = _frontier_frame()
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


def test_frontier_evaluate_rejects_invalid_input(tmp_path) -> None:
    frame = _frontier_frame()
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
