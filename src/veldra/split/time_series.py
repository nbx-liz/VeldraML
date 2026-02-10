"""Time series splitter contracts for leakage-safe folds."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Literal

import numpy as np


class TimeSeriesSplitter:
    """Simple expanding-window splitter with optional gap."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int | None = None,
        gap: int = 0,
        embargo: int = 0,
        mode: Literal["expanding", "blocked"] = "expanding",
        train_size: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        if mode not in {"expanding", "blocked"}:
            raise ValueError("mode must be either 'expanding' or 'blocked'")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be >= 1 when provided")
        if train_size is not None and train_size < 1:
            raise ValueError("train_size must be >= 1 when provided")
        if mode == "blocked" and train_size is None:
            raise ValueError("train_size is required when mode='blocked'")
        if mode == "expanding" and train_size is not None:
            raise ValueError("train_size can only be set when mode='blocked'")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.embargo = embargo
        self.mode = mode
        self.train_size = train_size

    def split(self, data: Sequence[Any] | int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = data if isinstance(data, int) else len(data)
        if n_samples <= self.n_splits + 1:
            raise ValueError("n_samples is too small for the requested number of splits.")

        test_size = self.test_size or (n_samples // (self.n_splits + 1))
        if test_size < 1:
            raise ValueError("test_size must resolve to >= 1.")

        generated_any = False
        embargoed: set[int] = set()
        for fold in range(self.n_splits):
            if self.mode == "expanding":
                train_end = test_size * (fold + 1)
                train_start = 0
            else:
                train_end = (self.train_size or 0) + (fold * test_size)
                train_start = train_end - (self.train_size or 0)

            test_start = train_end + self.gap
            test_end = test_start + test_size
            if test_end > n_samples:
                break

            train_idx_full = np.arange(train_start, train_end, dtype=int)
            if embargoed:
                embargoed_arr = np.fromiter(embargoed, dtype=int)
                train_idx = train_idx_full[~np.isin(train_idx_full, embargoed_arr)]
            else:
                train_idx = train_idx_full
            test_idx = np.arange(test_start, test_end, dtype=int)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            generated_any = True
            yield train_idx, test_idx

            embargo_start = test_start
            embargo_end = min(n_samples, test_end + self.embargo)
            if embargo_end > embargo_start:
                embargoed.update(range(embargo_start, embargo_end))

        if not generated_any:
            raise ValueError(
                "No valid folds were generated. Increase n_samples or adjust test_size/gap."
            )


def iter_time_series_splits(
    data: Sequence[Any] | int,
    n_splits: int = 5,
    test_size: int | None = None,
    gap: int = 0,
    embargo: int = 0,
    mode: Literal["expanding", "blocked"] = "expanding",
    train_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplitter(
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        embargo=embargo,
        mode=mode,
        train_size=train_size,
    )
    return list(splitter.split(data))
