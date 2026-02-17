from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.api import tune
from veldra.config.models import RunConfig
from veldra.modeling import tuning


def _regression_frame(rows: int = 36, seed: int = 321) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.3, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _multiclass_frame(rows_per_class: int = 12, seed: int = 322) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks: list[pd.DataFrame] = []
    for idx, label in enumerate(["a", "b", "c"]):
        x1 = rng.normal(loc=idx * 1.2, scale=0.4, size=rows_per_class)
        x2 = rng.normal(loc=-idx * 1.1, scale=0.4, size=rows_per_class)
        chunks.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(chunks, ignore_index=True)


def test_objective_spec_supports_new_metrics() -> None:
    assert tuning._objective_spec("regression", "mape") == ("mape", "minimize")
    assert tuning._objective_spec("multiclass", "multi_logloss") == (
        "multi_logloss",
        "minimize",
    )
    assert tuning._objective_spec("multiclass", "multi_error") == ("multi_error", "minimize")


def test_tune_supports_mape_and_multiclass_aliases(tmp_path) -> None:
    reg_path = tmp_path / "reg.csv"
    mc_path = tmp_path / "mc.csv"
    _regression_frame().to_csv(reg_path, index=False)
    _multiclass_frame().to_csv(mc_path, index=False)

    reg_result = tune(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(reg_path), "target": "target"},
            "split": {"type": "kfold", "n_splits": 2, "seed": 1},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "mape"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert reg_result.metadata["metric_name"] == "mape"

    mc_result = tune(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(mc_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 1},
            "tuning": {"enabled": True, "n_trials": 1, "objective": "multi_error"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert mc_result.metadata["metric_name"] == "multi_error"


def test_build_trial_config_supports_train_prefixed_params(config_payload) -> None:
    config = RunConfig.model_validate(config_payload("regression"))
    tuned = tuning._build_trial_config(
        config,
        {
            "train.num_boost_round": 777,
            "train.num_leaves_ratio": 0.8,
            "learning_rate": 0.05,
        },
    )
    assert tuned.train.num_boost_round == 777
    assert tuned.train.num_leaves_ratio == 0.8
    assert tuned.train.lgb_params["learning_rate"] == 0.05
