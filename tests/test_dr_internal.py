from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.causal import dr as dr_mod
from veldra.config.models import RunConfig


def _config(*, task_type: str = "regression", with_causal: bool = True) -> RunConfig:
    payload: dict[str, object] = {
        "config_version": 1,
        "task": {"type": task_type},
        "data": {"path": "dummy.csv", "target": "outcome"},
        "split": {"type": "kfold", "n_splits": 2, "seed": 7},
        "export": {"artifact_dir": "artifacts"},
    }
    if with_causal:
        payload["causal"] = {"treatment_col": "treatment"}
    return RunConfig.model_validate(payload)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 0.4, 0.7, 0.9],
            "x2": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "treatment": [0, 0, 1, 0, 1, 1],
            "outcome": [1.0, 1.1, 2.0, 1.4, 2.5, 2.7],
        }
    )


def test_feature_frame_requires_causal_config() -> None:
    cfg = _config(with_causal=False)
    with pytest.raises(VeldraValidationError, match="causal config is required"):
        dr_mod._feature_frame(cfg, _base_frame())


def test_feature_frame_validates_required_columns() -> None:
    cfg = _config()
    with pytest.raises(VeldraValidationError, match="Target column 'outcome' was not found"):
        dr_mod._feature_frame(cfg, _base_frame().drop(columns=["outcome"]))
    with pytest.raises(VeldraValidationError, match="Treatment column 'treatment' was not found"):
        dr_mod._feature_frame(cfg, _base_frame().drop(columns=["treatment"]))


def test_feature_frame_validates_empty_and_null_inputs() -> None:
    cfg = _config()
    with pytest.raises(VeldraValidationError, match="Input data is empty"):
        dr_mod._feature_frame(cfg, _base_frame().iloc[0:0])

    frame = _base_frame().copy()
    frame.loc[0, "outcome"] = np.nan
    with pytest.raises(VeldraValidationError, match="must not contain null values"):
        dr_mod._feature_frame(cfg, frame)


def test_feature_frame_validates_treatment_and_outcome_types() -> None:
    cfg = _config()
    bad_treatment = _base_frame().copy()
    bad_treatment["treatment"] = ["a", "b", "a", "b", "a", "b"]
    with pytest.raises(VeldraValidationError, match="Treatment column must be binary"):
        dr_mod._feature_frame(cfg, bad_treatment)

    bad_outcome = _base_frame().copy()
    bad_outcome["outcome"] = ["x", "y", "x", "y", "x", "y"]
    with pytest.raises(VeldraValidationError, match="Outcome values must be numeric"):
        dr_mod._feature_frame(cfg, bad_outcome)


def test_feature_frame_binary_outcome_requires_two_classes() -> None:
    cfg = _config(task_type="binary")
    frame = _base_frame().copy()
    frame["outcome"] = [1, 1, 1, 1, 1, 1]
    with pytest.raises(VeldraValidationError, match="Binary outcome requires two classes"):
        dr_mod._feature_frame(cfg, frame)


def test_feature_frame_validates_feature_availability() -> None:
    cfg_no_features = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {
                "path": "dummy.csv",
                "target": "outcome",
                "drop_cols": ["x1", "x2"],
            },
            "split": {"type": "kfold", "n_splits": 2, "seed": 7},
            "causal": {"treatment_col": "treatment"},
            "export": {"artifact_dir": "artifacts"},
        }
    )
    with pytest.raises(VeldraValidationError, match="No feature columns remain"):
        dr_mod._feature_frame(cfg_no_features, _base_frame())

    cfg = _config()
    frame = _base_frame().copy()
    frame["all_nan_feature"] = pd.Series([None] * len(frame), dtype="object")
    frame = frame[["all_nan_feature", "treatment", "outcome"]]
    with pytest.raises(
        VeldraValidationError,
        match="No usable feature columns remain after encoding",
    ):
        dr_mod._feature_frame(cfg, frame)


def test_nuisance_params_without_causal_returns_empty() -> None:
    cfg = _config(with_causal=False)
    assert dr_mod._nuisance_params(cfg, "propensity") == {}


def test_fit_outcome_model_rejects_empty_training_data() -> None:
    with pytest.raises(VeldraValidationError, match="Outcome model training set is empty"):
        dr_mod._fit_outcome_model(
            pd.DataFrame({"x1": []}),
            pd.Series([], dtype=float),
            seed=7,
            params={},
        )


def test_fit_calibrator_rejects_unsupported_method() -> None:
    with pytest.raises(VeldraValidationError, match="Unsupported propensity calibration method"):
        dr_mod._fit_calibrator("unknown", np.array([0.2, 0.8]), np.array([0, 1]), seed=7)


def test_run_dr_estimation_validates_basic_constraints() -> None:
    frame = _base_frame()
    cfg_without_causal = _config(with_causal=False)
    with pytest.raises(VeldraValidationError, match="causal config is required"):
        dr_mod.run_dr_estimation(cfg_without_causal, frame)

    cfg_wrong_method = _config()
    assert cfg_wrong_method.causal is not None
    object.__setattr__(cfg_wrong_method.causal, "method", "other")
    with pytest.raises(VeldraValidationError, match="Unsupported causal method 'other'"):
        dr_mod.run_dr_estimation(cfg_wrong_method, frame)

    with pytest.raises(VeldraValidationError, match="requires at least 4 rows"):
        dr_mod.run_dr_estimation(_config(), frame.iloc[:3])

    x, y, _ = dr_mod._feature_frame(_config(), frame)
    t_all_one = pd.Series(np.ones(len(x), dtype=int))

    def _fake_feature_frame(config: RunConfig, in_frame: pd.DataFrame):  # type: ignore[no-untyped-def]
        _ = config, in_frame
        return x, y, t_all_one

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dr_mod, "_feature_frame", _fake_feature_frame)
    with pytest.raises(
        VeldraValidationError,
        match="Treatment must include both treated and control",
    ):
        dr_mod.run_dr_estimation(_config(), frame)
    monkeypatch.undo()


def test_run_dr_estimation_non_crossfit_branch_executes() -> None:
    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "outcome"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 11},
            "causal": {"treatment_col": "treatment", "cross_fit": False},
            "export": {"artifact_dir": "artifacts"},
        }
    )
    result = dr_mod.run_dr_estimation(cfg, _base_frame())
    assert result.method == "dr"
    assert "dr" in result.metrics
    assert not result.observation_table.empty


def test_run_dr_estimation_detects_nan_nuisance_predictions(monkeypatch) -> None:
    cfg = _config()

    def _nan_propensity(model, x):  # type: ignore[no-untyped-def]
        return np.full(len(x), np.nan, dtype=float)

    monkeypatch.setattr(dr_mod, "_predict_propensity", _nan_propensity)
    with pytest.raises(
        VeldraValidationError,
        match="Failed to produce complete nuisance predictions",
    ):
        dr_mod.run_dr_estimation(cfg, _base_frame())
