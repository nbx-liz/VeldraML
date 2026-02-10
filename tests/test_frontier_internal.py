from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.config.models import RunConfig
from veldra.modeling import frontier


def _config(split_type: str = "kfold", *, path: str = "dummy.csv") -> RunConfig:
    payload: dict = {
        "config_version": 1,
        "task": {"type": "frontier"},
        "data": {"path": path, "target": "target"},
        "split": {"type": split_type, "n_splits": 2, "seed": 7},
        "frontier": {"alpha": 0.9},
    }
    if split_type == "group":
        payload["split"]["group_col"] = "group"
    if split_type == "timeseries":
        payload["split"]["time_col"] = "time_col"
    return RunConfig.model_validate(payload)


def _frame(rows: int = 8, seed: int = 55) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    target = 1.2 + 0.8 * x1 - 0.4 * x2 + rng.normal(scale=0.1, size=rows)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "target": target,
            "group": np.where(np.arange(rows) % 2 == 0, "g1", "g2"),
            "time_col": np.arange(rows),
        }
    )


@dataclass
class _FakeBooster:
    pred: np.ndarray
    best_iteration: int = 1

    def predict(self, _: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        _ = num_iteration
        return self.pred

    def model_to_string(self) -> str:
        return "frontier-model"


def test_build_feature_frame_validation_errors() -> None:
    cfg = _config()
    with pytest.raises(VeldraValidationError):
        frontier._build_feature_frame(cfg, pd.DataFrame({"x1": [1.0]}))
    with pytest.raises(VeldraValidationError):
        frontier._build_feature_frame(cfg, pd.DataFrame(columns=["x1", "target"]))

    bad = pd.DataFrame({"x1": [1.0], "target": [np.nan]})
    with pytest.raises(VeldraValidationError):
        frontier._build_feature_frame(cfg, bad)

    drop_all_cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": "dummy.csv", "target": "target", "drop_cols": ["x1", "x2"]},
        }
    )
    with pytest.raises(VeldraValidationError):
        frontier._build_feature_frame(
            drop_all_cfg,
            pd.DataFrame({"x1": [1.0], "x2": [2.0], "target": [3.0]}),
        )


def test_iter_cv_splits_branches_and_errors() -> None:
    df = _frame()
    x = df[["x1", "x2"]]
    assert frontier._iter_cv_splits(_config("kfold"), df, x)
    assert frontier._iter_cv_splits(_config("group"), df, x)
    assert frontier._iter_cv_splits(_config("timeseries"), df, x)

    bad_group = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "frontier"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "group", "n_splits": 2, "seed": 7, "group_col": "missing"},
        }
    )
    with pytest.raises(VeldraValidationError):
        frontier._iter_cv_splits(bad_group, df, x)

    cfg_stratified = _config("kfold")
    cfg_stratified.split.type = "stratified"  # type: ignore[assignment]
    with pytest.raises(VeldraValidationError):
        frontier._iter_cv_splits(cfg_stratified, df, x)

    unknown = _config("kfold")
    unknown.split.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(VeldraValidationError):
        frontier._iter_cv_splits(unknown, df, x)


def test_train_frontier_input_guardrails() -> None:
    not_frontier = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target"},
        }
    )
    with pytest.raises(VeldraValidationError):
        frontier.train_frontier_with_cv(not_frontier, _frame())

    cfg_no_path = _config(path=None)  # type: ignore[arg-type]
    with pytest.raises(VeldraValidationError):
        frontier.train_frontier_with_cv(cfg_no_path, _frame())


def test_train_frontier_empty_split_and_oof_missing(monkeypatch) -> None:
    cfg = _config()
    df = _frame()

    monkeypatch.setattr(
        frontier,
        "_iter_cv_splits",
        lambda c, d, x: [(np.array([], dtype=int), np.array([0], dtype=int))],
    )
    with pytest.raises(VeldraValidationError, match="empty train/valid split"):
        frontier.train_frontier_with_cv(cfg, df)

    monkeypatch.setattr(
        frontier,
        "_iter_cv_splits",
        lambda c, d, x: [(np.array([0, 1, 2], dtype=int), np.array([0, 1], dtype=int))],
    )
    monkeypatch.setattr(
        frontier,
        "_train_single_booster",
        lambda **kwargs: _FakeBooster(np.array([0.1, 0.2])),
    )
    with pytest.raises(VeldraValidationError, match="OOF predictions contain missing values"):
        frontier.train_frontier_with_cv(cfg, df)


def test_train_frontier_timeseries_sort_branch(monkeypatch) -> None:
    cfg = _config("timeseries")
    df = _frame().sample(frac=1.0, random_state=11).reset_index(drop=True)
    sorted_time = np.sort(df["time_col"].to_numpy())

    captured: dict[str, np.ndarray] = {}

    def _fake_build_feature_frame(config, data):  # type: ignore[no-untyped-def]
        captured["times"] = data["time_col"].to_numpy()
        return data[["x1", "x2"]], data["target"]

    monkeypatch.setattr(frontier, "_build_feature_frame", _fake_build_feature_frame)
    monkeypatch.setattr(
        frontier,
        "_iter_cv_splits",
        lambda c, d, x: [
            (np.array([4, 5, 6, 7], dtype=int), np.array([0, 1, 2, 3], dtype=int)),
            (np.array([0, 1, 2, 3], dtype=int), np.array([4, 5, 6, 7], dtype=int)),
        ],
    )

    def _fake_booster(**kwargs):  # type: ignore[no-untyped-def]
        return _FakeBooster(np.full(len(kwargs["x_valid"]), 1.23, dtype=float))

    monkeypatch.setattr(frontier, "_train_single_booster", _fake_booster)

    result = frontier.train_frontier_with_cv(cfg, df)
    assert np.array_equal(captured["times"], sorted_time)
    assert result.metrics["mean"]["mae"] >= 0.0


def test_frontier_timeseries_splitter_receives_extended_params(monkeypatch) -> None:
    cfg = _config("timeseries")
    cfg.split.timeseries_mode = "blocked"
    cfg.split.test_size = 2
    cfg.split.gap = 1
    cfg.split.embargo = 2
    cfg.split.train_size = 3

    df = _frame()
    x = df[["x1", "x2"]]
    captured: dict[str, int | str | None] = {}

    class _FakeSplitter:
        def __init__(
            self,
            n_splits: int,
            test_size: int | None,
            gap: int,
            embargo: int,
            mode: str,
            train_size: int | None,
        ) -> None:
            captured["n_splits"] = n_splits
            captured["test_size"] = test_size
            captured["gap"] = gap
            captured["embargo"] = embargo
            captured["mode"] = mode
            captured["train_size"] = train_size

        def split(self, data: int) -> list[tuple[np.ndarray, np.ndarray]]:
            _ = data
            return [(np.array([0, 1, 2], dtype=int), np.array([3, 4], dtype=int))]

    monkeypatch.setattr(frontier, "TimeSeriesSplitter", _FakeSplitter)
    splits = frontier._iter_cv_splits(cfg, df, x)
    assert splits
    assert captured == {
        "n_splits": cfg.split.n_splits,
        "test_size": 2,
        "gap": 1,
        "embargo": 2,
        "mode": "blocked",
        "train_size": 3,
    }
