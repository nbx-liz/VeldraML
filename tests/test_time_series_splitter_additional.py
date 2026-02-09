import pytest

from veldra.split import TimeSeriesSplitter, iter_time_series_splits


def test_timeseries_splitter_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=0)
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=2, gap=-1)


def test_timeseries_splitter_rejects_zero_test_size() -> None:
    splitter = TimeSeriesSplitter(n_splits=2, test_size=-1)
    with pytest.raises(ValueError):
        list(splitter.split(12))


def test_timeseries_splitter_raises_when_no_valid_fold_generated() -> None:
    splitter = TimeSeriesSplitter(n_splits=1, test_size=5, gap=10)
    with pytest.raises(ValueError):
        list(splitter.split(12))


def test_iter_time_series_splits_accepts_sequence_input() -> None:
    data = list(range(20))
    splits = iter_time_series_splits(data, n_splits=2, test_size=4, gap=1)
    assert len(splits) == 2
