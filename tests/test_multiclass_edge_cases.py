from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from veldra.api import fit
from veldra.api.exceptions import VeldraValidationError


def test_multiclass_fit_rejects_two_classes(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 1.1, 1.2, 2.1, 2.2],
            "x2": [1.0, 0.9, 0.2, 0.1, -0.3, -0.4],
            "target": ["a", "a", "b", "b", "a", "b"],
        }
    )
    path = tmp_path / "multiclass_two_class.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError, match="at least three target classes"):
        fit(
            {
                "config_version": 1,
                "task": {"type": "multiclass"},
                "data": {"path": str(path), "target": "target"},
                "split": {"type": "stratified", "n_splits": 2, "seed": 51},
                "export": {"artifact_dir": str(tmp_path / "artifacts")},
            }
        )


def test_multiclass_fit_with_many_classes(tmp_path) -> None:
    rng = np.random.default_rng(802)
    labels = [f"class_{idx:02d}" for idx in range(10)]
    rows: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        n = 10
        rows.append(
            pd.DataFrame(
                {
                    "x1": rng.normal(loc=idx * 0.4, scale=0.2, size=n),
                    "x2": rng.normal(loc=-idx * 0.3, scale=0.2, size=n),
                    "target": [label] * n,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    path = tmp_path / "multiclass_many_classes.csv"
    frame.to_csv(path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 52},
            "train": {"num_boost_round": 50, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert {"accuracy", "macro_f1", "logloss"} <= set(run.metrics)


def test_multiclass_low_frequency_class_with_stratified_split_is_supported(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 17),
            "x2": np.linspace(1.0, 0.0, 17),
            "target": ["a"] * 8 + ["b"] * 8 + ["rare"],
        }
    )
    path = tmp_path / "multiclass_low_frequency.csv"
    frame.to_csv(path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "multiclass"},
            "data": {"path": str(path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 53},
            "train": {"num_boost_round": 45, "early_stopping_rounds": 8},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )
    assert {"accuracy", "macro_f1", "logloss"} <= set(run.metrics)
