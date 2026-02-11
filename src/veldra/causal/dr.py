"""Doubly robust causal estimation (single-period MVP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig


@dataclass(slots=True)
class DREstimationOutput:
    method: str
    estimand: str
    estimate: float
    std_error: float | None
    ci_lower: float | None
    ci_upper: float | None
    metrics: dict[str, float]
    observation_table: pd.DataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class _ConstantModel:
    value: float

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


def _feature_frame(
    config: RunConfig, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if config.causal is None:
        raise VeldraValidationError("causal config is required for DR estimation.")
    target_col = config.data.target
    treatment_col = config.causal.treatment_col

    if target_col not in frame.columns:
        raise VeldraValidationError(f"Target column '{target_col}' was not found in input data.")
    if treatment_col not in frame.columns:
        raise VeldraValidationError(
            f"Treatment column '{treatment_col}' was not found in input data."
        )
    if frame.empty:
        raise VeldraValidationError("Input data is empty.")

    y = frame[target_col].copy()
    t_raw = frame[treatment_col].copy()
    if y.isna().any() or t_raw.isna().any():
        raise VeldraValidationError("Target/treatment columns must not contain null values.")

    t_num = pd.to_numeric(t_raw, errors="coerce")
    if t_num.isna().any():
        raise VeldraValidationError("Treatment column must be binary (0/1).")
    uniques = sorted(pd.unique(t_num).tolist())
    if len(uniques) != 2 or not set(uniques).issubset({0, 1}):
        raise VeldraValidationError(
            f"Treatment column must contain exactly two binary values (0/1), got {uniques}."
        )
    t = t_num.astype(int)

    if config.task.type == "binary":
        y_unique = pd.unique(y)
        if len(y_unique) != 2:
            raise VeldraValidationError(
                f"Binary outcome requires two classes, got {len(y_unique)}."
            )
        classes = sorted((v.item() if hasattr(v, "item") else v for v in y_unique), key=str)
        y = y.map({classes[0]: 0.0, classes[1]: 1.0})
        if y.isna().any():
            raise VeldraValidationError("Failed to map binary outcome to numeric labels.")
    else:
        try:
            y = pd.to_numeric(y, errors="raise").astype(float)
        except Exception as exc:
            raise VeldraValidationError(
                "Outcome values must be numeric for DR regression."
            ) from exc

    drop_cols = set(config.data.drop_cols + config.data.id_cols + [target_col, treatment_col])
    feature_cols = [c for c in frame.columns if c not in drop_cols]
    if not feature_cols:
        raise VeldraValidationError("No feature columns remain for DR estimation.")
    x = pd.get_dummies(frame.loc[:, feature_cols], drop_first=False)
    if x.empty:
        raise VeldraValidationError("No usable feature columns remain after encoding.")
    return x.astype(float), y.astype(float), t


def _nuisance_params(config: RunConfig, key: str) -> dict[str, Any]:
    if config.causal is None:
        return {}
    raw = config.causal.nuisance_params.get(key, {})
    return raw if isinstance(raw, dict) else {}


def _fit_propensity_model(
    x: pd.DataFrame,
    t: pd.Series,
    seed: int,
    params: dict[str, Any],
) -> lgb.LGBMClassifier | _ConstantModel:
    rate = float(t.mean())
    if rate <= 0.0 or rate >= 1.0:
        return _ConstantModel(rate)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
        **params,
    )
    model.fit(x, t)
    return model


def _predict_propensity(model: lgb.LGBMClassifier | _ConstantModel, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, _ConstantModel):
        return model.predict(x)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def _fit_outcome_model(
    x: pd.DataFrame,
    y: pd.Series,
    seed: int,
    params: dict[str, Any],
) -> lgb.LGBMRegressor | _ConstantModel:
    if len(y) == 0:
        raise VeldraValidationError("Outcome model training set is empty.")
    if np.isclose(float(y.std(ddof=0)), 0.0):
        return _ConstantModel(float(y.iloc[0]))
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
        **params,
    )
    model.fit(x, y)
    return model


def _fit_outcome_with_fallback(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    subgroup_mask: pd.Series,
    seed: int,
    params: dict[str, Any],
) -> lgb.LGBMRegressor | _ConstantModel:
    if subgroup_mask.sum() == 0:
        return _ConstantModel(float(y_train.mean()))
    return _fit_outcome_model(
        x_train.loc[subgroup_mask],
        y_train.loc[subgroup_mask],
        seed,
        params,
    )


def _predict_outcome(model: lgb.LGBMRegressor | _ConstantModel, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, _ConstantModel):
        return model.predict(x)
    return np.asarray(model.predict(x), dtype=float)


def _fit_calibrator(
    method: str,
    e_raw: np.ndarray,
    t: np.ndarray,
    seed: int,
) -> LogisticRegression | IsotonicRegression:
    e_for_fit = np.clip(e_raw, 1e-6, 1.0 - 1e-6)
    if method == "platt":
        cal = LogisticRegression(random_state=seed, max_iter=1000)
        cal.fit(e_for_fit.reshape(-1, 1), t)
        return cal
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(e_for_fit, t)
        return cal
    raise VeldraValidationError(f"Unsupported propensity calibration method '{method}'.")


def _att_score(y: np.ndarray, t: np.ndarray, e: np.ndarray, m0: np.ndarray) -> np.ndarray:
    p1 = float(np.mean(t))
    core = t * (y - m0) - (1.0 - t) * (e / (1.0 - e)) * (y - m0)
    return core / p1


def _ate_score(
    y: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    m1: np.ndarray,
    m0: np.ndarray,
) -> np.ndarray:
    return m1 - m0 + t * (y - m1) / e - (1.0 - t) * (y - m0) / (1.0 - e)


def run_dr_estimation(config: RunConfig, frame: pd.DataFrame) -> DREstimationOutput:
    """Run doubly robust estimation using cross-fitting and calibrated propensity."""
    if config.causal is None:
        raise VeldraValidationError("causal config is required for DR estimation.")
    if config.causal.method != "dr":
        raise VeldraValidationError(f"Unsupported causal method '{config.causal.method}'.")

    x, y, t = _feature_frame(config, frame)
    n_rows = len(x)
    if n_rows < 4:
        raise VeldraValidationError("DR estimation requires at least 4 rows.")
    if t.nunique() != 2:
        raise VeldraValidationError("Treatment must include both treated and control observations.")

    propensity_params = _nuisance_params(config, "propensity")
    outcome_params = _nuisance_params(config, "outcome")

    e_raw = np.full(n_rows, np.nan, dtype=float)
    m1_hat = np.full(n_rows, np.nan, dtype=float)
    m0_hat = np.full(n_rows, np.nan, dtype=float)

    if config.causal.cross_fit:
        n_splits = min(max(2, config.split.n_splits), n_rows)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=config.split.seed)
        for train_idx, valid_idx in splitter.split(x):
            x_train = x.iloc[train_idx]
            x_valid = x.iloc[valid_idx]
            t_train = t.iloc[train_idx]
            y_train = y.iloc[train_idx]

            prop_model = _fit_propensity_model(
                x_train, t_train, config.train.seed, propensity_params
            )
            e_raw[valid_idx] = _predict_propensity(prop_model, x_valid)

            treated_mask = t_train == 1
            control_mask = t_train == 0
            m1_model = _fit_outcome_with_fallback(
                x_train,
                y_train,
                treated_mask,
                config.train.seed,
                outcome_params,
            )
            m0_model = _fit_outcome_with_fallback(
                x_train,
                y_train,
                control_mask,
                config.train.seed,
                outcome_params,
            )
            m1_hat[valid_idx] = _predict_outcome(m1_model, x_valid)
            m0_hat[valid_idx] = _predict_outcome(m0_model, x_valid)
    else:
        prop_model = _fit_propensity_model(x, t, config.train.seed, propensity_params)
        e_raw[:] = _predict_propensity(prop_model, x)
        m1_model = _fit_outcome_model(
            x.loc[t == 1], y.loc[t == 1], config.train.seed, outcome_params
        )
        m0_model = _fit_outcome_model(
            x.loc[t == 0], y.loc[t == 0], config.train.seed, outcome_params
        )
        m1_hat[:] = _predict_outcome(m1_model, x)
        m0_hat[:] = _predict_outcome(m0_model, x)

    if np.isnan(e_raw).any() or np.isnan(m1_hat).any() or np.isnan(m0_hat).any():
        raise VeldraValidationError("Failed to produce complete nuisance predictions for DR.")

    cal = _fit_calibrator(
        config.causal.propensity_calibration,
        e_raw,
        t.to_numpy(dtype=int),
        config.train.seed,
    )
    if isinstance(cal, LogisticRegression):
        e_hat = np.asarray(
            cal.predict_proba(np.clip(e_raw, 1e-6, 1 - 1e-6).reshape(-1, 1))[:, 1],
            dtype=float,
        )
    else:
        e_hat = np.asarray(
            cal.predict(np.clip(e_raw, 1e-6, 1 - 1e-6)),
            dtype=float,
        )
    e_hat = np.clip(e_hat, config.causal.propensity_clip, 1.0 - config.causal.propensity_clip)

    y_np = y.to_numpy(dtype=float)
    t_np = t.to_numpy(dtype=float)
    m1_np = np.asarray(m1_hat, dtype=float)
    m0_np = np.asarray(m0_hat, dtype=float)

    if config.task.type == "binary":
        m1_np = np.clip(m1_np, 0.0, 1.0)
        m0_np = np.clip(m0_np, 0.0, 1.0)

    estimand = config.causal.estimand
    if estimand == "ate":
        score = _ate_score(y_np, t_np, e_hat, m1_np, m0_np)
        ipw = float(np.mean(t_np * y_np / e_hat - (1.0 - t_np) * y_np / (1.0 - e_hat)))
        weights = t_np / e_hat - (1.0 - t_np) / (1.0 - e_hat)
    else:
        score = _att_score(y_np, t_np, e_hat, m0_np)
        p1 = float(np.mean(t_np))
        ipw = float(
            np.mean((t_np * y_np - (1.0 - t_np) * (e_hat / (1.0 - e_hat)) * y_np) / p1)
        )
        weights = t_np / p1 - (1.0 - t_np) * (e_hat / (1.0 - e_hat)) / p1

    estimate = float(np.mean(score))
    std_error = float(np.std(score, ddof=1) / np.sqrt(len(score))) if len(score) > 1 else None
    ci_lower = float(estimate - 1.96 * std_error) if std_error is not None else None
    ci_upper = float(estimate + 1.96 * std_error) if std_error is not None else None
    naive = float(np.mean(y_np[t_np == 1.0]) - np.mean(y_np[t_np == 0.0]))

    observation = pd.DataFrame(
        {
            "treatment": t_np.astype(int),
            "outcome": y_np,
            "e_raw": e_raw,
            "e_hat": e_hat,
            "m1_hat": m1_np,
            "m0_hat": m0_np,
            "psi": score,
            "weight": weights,
        }
    )
    metrics = {
        "naive": naive,
        "ipw": ipw,
        "dr": estimate,
        "treated_rate": float(np.mean(t_np)),
        "coverage_treated": float(np.mean(e_hat[t_np == 1.0])),
    }
    summary: dict[str, Any] = {
        "method": config.causal.method,
        "estimand": estimand,
        "estimate": estimate,
        "std_error": std_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "metrics": metrics,
        "n_rows": int(n_rows),
        "n_treated": int(np.sum(t_np == 1.0)),
        "n_control": int(np.sum(t_np == 0.0)),
        "propensity_clip": float(config.causal.propensity_clip),
        "propensity_calibration": config.causal.propensity_calibration,
        "cross_fit": bool(config.causal.cross_fit),
    }

    return DREstimationOutput(
        method=config.causal.method,
        estimand=estimand,
        estimate=estimate,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        metrics=metrics,
        observation_table=observation,
        summary=summary,
    )
