from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.diagnostics.metrics import (
    binary_metrics,
    frontier_metrics,
    multiclass_metrics,
    regression_metrics,
    split_in_out_metrics,
)


def test_regression_metrics_contract() -> None:
    out = regression_metrics([1.0, 2.0, 3.0], [1.1, 1.8, 3.2])
    assert {"rmse", "mae", "mape", "r2"} <= set(out)


def test_binary_metrics_contract() -> None:
    out = binary_metrics([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.7])
    assert 0.0 <= float(out["auc"]) <= 1.0
    assert 0.0 <= float(out["brier"]) <= 1.0


def test_multiclass_metrics_contract() -> None:
    y = np.array([0, 1, 2, 1])
    proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.1, 0.6, 0.3],
        ]
    )
    out = multiclass_metrics(y, proba)
    assert "multi_logloss" in out
    assert "multi_error" in out


def test_frontier_metrics_contract() -> None:
    out = frontier_metrics([1.0, 2.0, 3.0], [1.2, 2.1, 2.9], alpha=0.9)
    assert {"pinball", "mae", "coverage"} <= set(out)


def test_split_in_out_metrics_returns_two_labels() -> None:
    out = split_in_out_metrics(
        regression_metrics,
        y_true=[1.0, 2.0, 3.0, 4.0],
        y_pred=[1.1, 2.2, 2.9, 4.1],
        fold_ids=[0, 0, 1, 1],
        eval_fold_ids={1},
    )
    assert isinstance(out, pd.DataFrame)
    assert list(out["label"]) == ["in_sample", "out_of_sample"]
    assert {"rmse", "mae", "mape", "r2"} <= set(out.columns)
