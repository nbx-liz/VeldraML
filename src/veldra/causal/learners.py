"""Nuisance learner interfaces and default factories for causal estimation."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd


class PropensityLearner(Protocol):
    """Protocol for treatment propensity estimators."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


class OutcomeLearner(Protocol):
    """Protocol for outcome estimators."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict(self, x: pd.DataFrame) -> np.ndarray: ...


PropensityFactory = Callable[[int, dict[str, Any]], PropensityLearner]
OutcomeFactory = Callable[[int, dict[str, Any]], OutcomeLearner]


def default_propensity_factory(seed: int, params: dict[str, Any]) -> PropensityLearner:
    """Build default LightGBM propensity learner."""
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
        **params,
    )


def default_outcome_factory(seed: int, params: dict[str, Any]) -> OutcomeLearner:
    """Build default LightGBM outcome learner."""
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
        **params,
    )

