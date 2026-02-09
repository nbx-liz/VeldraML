import numpy as np
import pytest

from veldra.split import TimeSeriesSplitter


def test_timeseries_splitter_is_ordered_and_non_overlapping() -> None:
    splitter = TimeSeriesSplitter(n_splits=3, test_size=5, gap=1)
    splits = list(splitter.split(30))

    assert len(splits) == 3

    for train_idx, test_idx in splits:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert train_idx.max() < test_idx.min()
        assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())


def test_timeseries_splitter_rejects_too_small_dataset() -> None:
    splitter = TimeSeriesSplitter(n_splits=3, test_size=2)

    with pytest.raises(ValueError):
        list(splitter.split(4))
