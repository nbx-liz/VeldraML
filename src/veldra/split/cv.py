"""Task-agnostic CV split helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.split.time_series import TimeSeriesSplitter


def iter_cv_splits(
    config: RunConfig,
    data: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build CV splits from ``RunConfig.split``."""
    split_cfg = config.split
    if split_cfg.type == "stratified":
        if y is None:
            raise VeldraValidationError("split.type='stratified' requires target labels.")
        splitter = StratifiedKFold(
            n_splits=split_cfg.n_splits,
            shuffle=True,
            random_state=split_cfg.seed,
        )
        return list(splitter.split(x, y))

    if split_cfg.type == "kfold":
        splitter = KFold(
            n_splits=split_cfg.n_splits,
            shuffle=True,
            random_state=split_cfg.seed,
        )
        return list(splitter.split(x))

    if split_cfg.type == "group":
        if not split_cfg.group_col:
            raise VeldraValidationError("split.group_col is required for group split.")
        if split_cfg.group_col not in data.columns:
            raise VeldraValidationError(
                f"Group column '{split_cfg.group_col}' was not found in input data."
            )
        splitter = GroupKFold(n_splits=split_cfg.n_splits)
        if y is None:
            return list(splitter.split(x, groups=data[split_cfg.group_col]))
        return list(splitter.split(x, y, groups=data[split_cfg.group_col]))

    if split_cfg.type == "timeseries":
        if not split_cfg.time_col:
            raise VeldraValidationError("split.time_col is required for timeseries split.")
        ordered = data.sort_values(split_cfg.time_col).reset_index(drop=True)
        x_ordered = ordered.loc[:, x.columns]
        splitter = TimeSeriesSplitter(
            n_splits=split_cfg.n_splits,
            test_size=split_cfg.test_size,
            gap=split_cfg.gap,
            embargo=split_cfg.embargo,
            mode=split_cfg.timeseries_mode,
            train_size=split_cfg.train_size,
        )
        return list(splitter.split(len(x_ordered)))

    raise VeldraValidationError(f"Unsupported split type '{split_cfg.type}'.")
