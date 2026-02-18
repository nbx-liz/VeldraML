import pytest

from veldra.split import TimeSeriesSplitter, iter_time_series_splits


def test_timeseries_splitter_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=0)
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=2, gap=-1)
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=2, embargo=-1)
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=2, mode="blocked")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=2, mode="expanding", train_size=10)


def test_timeseries_splitter_rejects_zero_test_size() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSplitter(n_splits=2, test_size=-1)


def test_timeseries_splitter_raises_when_no_valid_fold_generated() -> None:
    splitter = TimeSeriesSplitter(n_splits=1, test_size=5, gap=10)
    with pytest.raises(ValueError):
        list(splitter.split(12))


def test_iter_time_series_splits_accepts_sequence_input() -> None:
    data = list(range(20))
    splits = iter_time_series_splits(
        data,
        n_splits=2,
        test_size=4,
        gap=1,
        embargo=1,
        mode="expanding",
    )
    assert len(splits) == 2


def test_timeseries_splitter_blocked_mode_uses_fixed_train_size() -> None:
    splitter = TimeSeriesSplitter(
        n_splits=2,
        mode="blocked",
        train_size=4,
        test_size=2,
    )
    splits = list(splitter.split(12))
    assert len(splits) == 2
    assert len(splits[0][0]) == 4
    assert len(splits[0][1]) == 2
    assert len(splits[1][0]) <= 4
    assert len(splits[1][1]) == 2
    prior_test = set(splits[0][1].tolist())
    assert prior_test.isdisjoint(set(splits[1][0].tolist()))


def test_timeseries_splitter_embargo_excludes_previous_test_from_future_train() -> None:
    splitter = TimeSeriesSplitter(
        n_splits=2,
        mode="expanding",
        test_size=3,
        gap=0,
        embargo=2,
    )
    splits = list(splitter.split(15))
    assert len(splits) == 2
    first_test = set(splits[0][1].tolist())
    second_train = set(splits[1][0].tolist())
    assert first_test.isdisjoint(second_train)


def test_timeseries_splitter_supports_single_period_split() -> None:
    splitter = TimeSeriesSplitter(n_splits=1, test_size=1, mode="expanding")
    splits = list(splitter.split(4))
    assert len(splits) == 1
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 1
    assert len(test_idx) == 1


def test_timeseries_splitter_rejects_insufficient_periods() -> None:
    splitter = TimeSeriesSplitter(n_splits=3, test_size=1, mode="expanding")
    with pytest.raises(ValueError, match="n_samples is too small"):
        list(splitter.split(4))
