from __future__ import annotations

import pandas as pd

from veldra.api import fit
from veldra.api.artifact import Artifact
from veldra.config.models import RunConfig
from veldra.modeling.binary import train_binary_with_cv
from veldra.modeling.frontier import train_frontier_with_cv
from veldra.modeling.multiclass import train_multiclass_with_cv
from veldra.modeling.regression import train_regression_with_cv


def _regression_config(path: str, artifact_dir: str) -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": path, "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 1},
            "export": {"artifact_dir": artifact_dir},
        }
    )


def test_training_outputs_include_observation_table(
    tmp_path,
    regression_frame,
    binary_frame,
    multiclass_frame,
    frontier_frame,
) -> None:
    reg_df = regression_frame(rows=36)
    bin_df = binary_frame(rows=36)
    mc_df = multiclass_frame(rows_per_class=12)
    fr_df = frontier_frame(rows=36)

    reg_cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(tmp_path / "reg.csv"), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 1},
        }
    )
    bin_cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(tmp_path / "bin.csv"), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 1},
            "postprocess": {"calibration": "platt"},
        }
    )
    mc_cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(tmp_path / "mc.csv"), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 1},
        }
    )
    fr_cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": str(tmp_path / "fr.csv"), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 1},
            "frontier": {"alpha": 0.9},
        }
    )

    reg_out = train_regression_with_cv(reg_cfg, reg_df)
    bin_out = train_binary_with_cv(bin_cfg, bin_df)
    mc_out = train_multiclass_with_cv(mc_cfg, mc_df)
    fr_out = train_frontier_with_cv(fr_cfg, fr_df)

    assert {"fold_id", "prediction", "residual"} <= set(reg_out.observation_table.columns)
    assert {"fold_id", "score", "score_raw"} <= set(bin_out.observation_table.columns)
    assert {"fold_id", "label_pred"} <= set(mc_out.observation_table.columns)
    assert {"fold_id", "prediction", "efficiency"} <= set(fr_out.observation_table.columns)


def test_artifact_roundtrip_preserves_observation_table(tmp_path, regression_frame) -> None:
    train_path = tmp_path / "train.csv"
    regression_frame(rows=40).to_csv(train_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(train_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 1},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    artifact = Artifact.load(run.artifact_path)
    assert isinstance(artifact.observation_table, pd.DataFrame)
    assert not artifact.observation_table.empty
    assert {"fold_id", "prediction", "residual"} <= set(artifact.observation_table.columns)
