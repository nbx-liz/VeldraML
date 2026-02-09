"""Time series splitter contracts for leakage-safe folds."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np


class TimeSeriesSplitter:
    """Simple expanding-window splitter with optional gap."""

    def __init__(self, n_splits: int = 5, test_size: int | None = None, gap: int = 0) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, data: Sequence[Any] | int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = data if isinstance(data, int) else len(data)
        if n_samples <= self.n_splits + 1:
            raise ValueError("n_samples is too small for the requested number of splits.")

        test_size = self.test_size or (n_samples // (self.n_splits + 1))
        if test_size < 1:
            raise ValueError("test_size must resolve to >= 1.")

        generated_any = False
        for fold in range(self.n_splits):
            train_end = test_size * (fold + 1)
            test_start = train_end + self.gap
            test_end = test_start + test_size
            if test_end > n_samples:
                break

            train_idx = np.arange(0, train_end, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            generated_any = True
            yield train_idx, test_idx

        if not generated_any:
            raise ValueError(
                "No valid folds were generated. Increase n_samples or adjust test_size/gap."
            )


def iter_time_series_splits(
    data: Sequence[Any] | int,
    n_splits: int = 5,
    test_size: int | None = None,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplitter(n_splits=n_splits, test_size=test_size, gap=gap)
    return list(splitter.split(data))
