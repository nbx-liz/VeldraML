"""Shared modeling helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig


def split_for_early_stopping(
    x: pd.DataFrame,
    y: pd.Series,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a validation split for early stopping from training rows.

    Notes
    -----
    - When early stopping is disabled, the original input is returned twice
      for backward-compatible train/valid handling.
    - For timeseries split, the most recent tail is used as validation.
    - For binary/multiclass, stratified shuffling is preferred and falls back
      to random shuffle when class counts are insufficient.
    """
    if config.train.early_stopping_rounds is None:
        return x, x, y, y

    if len(x) != len(y):
        raise VeldraValidationError("x and y length mismatch in early stopping split.")
    if len(x) < 2:
        raise VeldraValidationError(
            "Early stopping requires at least 2 training rows after fold split."
        )

    fraction = float(config.train.early_stopping_validation_fraction)
    n_valid = max(1, int(round(len(x) * fraction)))
    n_valid = min(n_valid, len(x) - 1)

    if config.split.type == "timeseries":
        cut = len(x) - n_valid
        if cut <= 0:
            raise VeldraValidationError(
                "Insufficient rows to create timeseries early-stopping split."
            )
        train_idx = list(range(0, cut))
        valid_idx = list(range(cut, len(x)))
        return x.iloc[train_idx], x.iloc[valid_idx], y.iloc[train_idx], y.iloc[valid_idx]

    if config.task.type in {"binary", "multiclass"}:
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=n_valid,
                random_state=config.train.seed,
            )
            train_idx, valid_idx = next(splitter.split(x, y))
            return (
                x.iloc[train_idx],
                x.iloc[valid_idx],
                y.iloc[train_idx],
                y.iloc[valid_idx],
            )
        except ValueError:
            # Fallback when tiny folds cannot satisfy stratification constraints.
            pass

    splitter = ShuffleSplit(
        n_splits=1,
        test_size=n_valid,
        random_state=config.train.seed,
    )
    train_idx, valid_idx = next(splitter.split(x))
    return x.iloc[train_idx], x.iloc[valid_idx], y.iloc[train_idx], y.iloc[valid_idx]
