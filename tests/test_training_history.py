from __future__ import annotations

from pathlib import Path

from veldra.api.artifact import Artifact
from veldra.config.models import RunConfig


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"target": "y"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )


def test_training_history_saved_and_loaded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    history = {
        "folds": [
            {
                "fold": 1,
                "best_iteration": 7,
                "num_iterations": 12,
                "eval_history": {"rmse": [0.8, 0.7]},
            }
        ],
        "final_model": {"best_iteration": 9, "num_iterations": 15, "eval_history": {"rmse": [0.6]}},
    }
    artifact = Artifact.from_config(
        config,
        run_id="run-hist",
        feature_schema={"feature_names": ["x1"], "target": "y", "task_type": "regression"},
        training_history=history,
    )
    out_dir = tmp_path / "artifact_hist"
    artifact.save(out_dir)

    loaded = Artifact.load(out_dir)
    assert loaded.training_history == history
    assert (out_dir / "training_history.json").exists()


def test_training_history_is_optional_for_legacy_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifact = Artifact.from_config(
        config,
        run_id="run-no-hist",
        feature_schema={"feature_names": ["x1"], "target": "y", "task_type": "regression"},
    )
    out_dir = tmp_path / "artifact_no_hist"
    artifact.save(out_dir)

    loaded = Artifact.load(out_dir)
    assert loaded.training_history is None
