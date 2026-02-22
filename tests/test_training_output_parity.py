from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from veldra.config.models import RunConfig
from veldra.modeling.binary import train_binary_with_cv
from veldra.modeling.frontier import train_frontier_with_cv
from veldra.modeling.multiclass import train_multiclass_with_cv
from veldra.modeling.regression import train_regression_with_cv

_BASELINE_DIR = Path("tests/fixtures/training_output_parity")
_COMMON_TRAIN = {"lgb_params": {"num_threads": 1}}


def _load_baseline(task_name: str) -> dict:
    with (_BASELINE_DIR / f"{task_name}.pkl").open("rb") as f:
        return pickle.load(f)


def _assert_training_history_parity(
    actual: dict,
    expected: dict,
    *,
    expected_rows: int,
) -> None:
    assert actual["folds"] == expected["folds"]
    assert actual["final_model"] == expected["final_model"]
    assert actual.get("oof_total_rows") == expected_rows
    assert actual.get("oof_scored_rows") == expected_rows
    assert actual.get("oof_coverage_ratio") == 1.0


def test_training_output_parity_regression() -> None:
    rng = np.random.default_rng(1201)
    rows = 48
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 3.0 + 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.3, size=rows)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 13},
            "train": _COMMON_TRAIN,
        }
    )

    out = train_regression_with_cv(cfg, frame)
    expected = _load_baseline("regression")
    assert out.model_text == expected["model_text"]
    assert out.metrics == expected["metrics"]
    assert out.feature_schema == expected["feature_schema"]
    _assert_training_history_parity(
        out.training_history,
        expected["training_history"],
        expected_rows=rows,
    )
    pdt.assert_frame_equal(out.cv_results, expected["cv_results"], check_exact=True)
    pdt.assert_frame_equal(out.observation_table, expected["observation_table"], check_exact=True)


def test_training_output_parity_binary() -> None:
    rng = np.random.default_rng(1202)
    rows = 60
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = 0.9 * x1 - 1.1 * x2 + rng.normal(scale=0.6, size=rows)
    y = (score > np.median(score)).astype(int)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 13},
            "postprocess": {"calibration": "platt"},
            "train": _COMMON_TRAIN,
        }
    )

    out = train_binary_with_cv(cfg, frame)
    expected = _load_baseline("binary")
    assert out.model_text == expected["model_text"]
    assert out.metrics == expected["metrics"]
    assert out.feature_schema == expected["feature_schema"]
    _assert_training_history_parity(
        out.training_history,
        expected["training_history"],
        expected_rows=rows,
    )
    assert out.threshold == expected["threshold"]
    pdt.assert_frame_equal(out.cv_results, expected["cv_results"], check_exact=True)
    pdt.assert_frame_equal(out.observation_table, expected["observation_table"], check_exact=True)
    pdt.assert_frame_equal(
        out.calibration_curve,
        expected["calibration_curve"],
        check_exact=True,
    )


def test_training_output_parity_multiclass() -> None:
    rng = np.random.default_rng(1203)
    rows_per_class = 18
    alpha = pd.DataFrame(
        {
            "x1": rng.normal(loc=-1.0, scale=0.4, size=rows_per_class),
            "x2": rng.normal(loc=1.0, scale=0.4, size=rows_per_class),
            "target": "alpha",
        }
    )
    beta = pd.DataFrame(
        {
            "x1": rng.normal(loc=1.0, scale=0.4, size=rows_per_class),
            "x2": rng.normal(loc=0.0, scale=0.4, size=rows_per_class),
            "target": "beta",
        }
    )
    gamma = pd.DataFrame(
        {
            "x1": rng.normal(loc=0.0, scale=0.4, size=rows_per_class),
            "x2": rng.normal(loc=-1.0, scale=0.4, size=rows_per_class),
            "target": "gamma",
        }
    )
    frame = pd.concat([alpha, beta, gamma], ignore_index=True)
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 13},
            "train": _COMMON_TRAIN,
        }
    )

    out = train_multiclass_with_cv(cfg, frame)
    expected = _load_baseline("multiclass")
    assert out.model_text == expected["model_text"]
    assert out.metrics == expected["metrics"]
    assert out.feature_schema == expected["feature_schema"]
    _assert_training_history_parity(
        out.training_history,
        expected["training_history"],
        expected_rows=len(frame),
    )
    pdt.assert_frame_equal(out.cv_results, expected["cv_results"], check_exact=True)
    pdt.assert_frame_equal(out.observation_table, expected["observation_table"], check_exact=True)


def test_training_output_parity_frontier() -> None:
    rng = np.random.default_rng(1204)
    rows = 54
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    base = 2.0 + 0.9 * x1 - 0.4 * x2
    noise = np.abs(rng.normal(scale=0.25, size=rows))
    y = base + noise
    frame = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 13},
            "frontier": {"alpha": 0.9},
            "train": _COMMON_TRAIN,
        }
    )

    out = train_frontier_with_cv(cfg, frame)
    expected = _load_baseline("frontier")
    assert out.model_text == expected["model_text"]
    assert out.metrics == expected["metrics"]
    assert out.feature_schema == expected["feature_schema"]
    _assert_training_history_parity(
        out.training_history,
        expected["training_history"],
        expected_rows=rows,
    )
    pdt.assert_frame_equal(out.cv_results, expected["cv_results"], check_exact=True)
    pdt.assert_frame_equal(out.observation_table, expected["observation_table"], check_exact=True)
