from __future__ import annotations

import numpy as np
import pandas as pd

from veldra.config.models import RunConfig
from veldra.modeling import binary


class _DummyBooster:
    best_iteration = 1

    def current_iteration(self) -> int:
        return 1

    def predict(self, x: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        _ = num_iteration
        return np.full(len(x), 0.5, dtype=float)

    def model_to_string(self) -> str:
        return "dummy"


def _binary_config(top_k: int = 5) -> RunConfig:
    return RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "stratified", "n_splits": 2, "seed": 7},
            "train": {"top_k": top_k, "num_boost_round": 8},
        }
    )


def test_precision_at_k_helper() -> None:
    y_true = np.array([1, 0, 1, 0, 1], dtype=int)
    y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.7], dtype=float)
    assert binary._precision_at_k(y_true, y_score, 2) == 0.5
    assert binary._precision_at_k(y_true, y_score, 10) == 0.6


def test_train_single_booster_uses_top_k_feval(monkeypatch) -> None:
    cfg = _binary_config(top_k=3)
    x = pd.DataFrame({"x1": [0.1, 0.2, 0.3, 0.4], "x2": [1.0, 0.7, 0.2, 1.2]})
    y = pd.Series([0, 1, 0, 1], name="target")
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):  # type: ignore[no-untyped-def]
        captured["params"] = kwargs["params"]
        captured["feval"] = kwargs["feval"]
        return _DummyBooster()

    monkeypatch.setattr(binary.lgb, "train", _fake_train)
    binary._train_single_booster(x, y, x, y, cfg)

    params = captured["params"]
    assert isinstance(params, dict)
    assert params["metric"] == "None"
    assert callable(captured["feval"])


def test_train_binary_with_cv_adds_precision_at_k_metric(binary_frame) -> None:
    cfg = _binary_config(top_k=4)
    frame = binary_frame(rows=80, seed=17, coef1=1.2, coef2=-0.8, noise=0.35)
    out = binary.train_binary_with_cv(cfg, frame)
    assert "precision_at_4" in out.metrics["mean"]
    assert "accuracy" in out.metrics["mean"]
    assert "precision" in out.metrics["mean"]
    assert "recall" in out.metrics["mean"]


def test_training_history_contains_precision_at_k(binary_frame) -> None:
    cfg = _binary_config(top_k=6)
    frame = binary_frame(rows=70, seed=19, coef1=1.3, coef2=-0.6, noise=0.4)
    out = binary.train_binary_with_cv(cfg, frame)
    assert out.training_history is not None
    fold_history = out.training_history["folds"][0]["eval_history"]
    assert "precision_at_6" in fold_history
