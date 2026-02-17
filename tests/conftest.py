"""Pytest shared setup."""

from __future__ import annotations

import shutil
import sys
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GUI_TEST_FILES = {
    "test_new_ux.py",
}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gui: GUI adapter related tests.")
    config.addinivalue_line("markers", "gui_e2e: Playwright GUI end-to-end tests.")
    config.addinivalue_line("markers", "gui_smoke: CI smoke subset of Playwright GUI tests.")
    config.addinivalue_line(
        "markers",
        "notebook_e2e: Notebook execution evidence and output contract tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _ = config
    for item in items:
        path = Path(str(item.fspath)).name
        if path.startswith("test_gui_") or path in GUI_TEST_FILES:
            item.add_marker(pytest.mark.gui)


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-local tmp_path to avoid permission issues in this environment."""
    temp_root = REPO_ROOT / ".pytest_tmp" / "cases"
    temp_root.mkdir(parents=True, exist_ok=True)
    created = temp_root / f"case_{uuid4().hex}"
    created.mkdir(parents=True, exist_ok=False)
    try:
        yield created
    finally:
        shutil.rmtree(created, ignore_errors=True)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@pytest.fixture
def binary_frame():
    def _build(
        rows: int = 120,
        seed: int = 11,
        coef1: float = 1.8,
        coef2: float = -1.2,
        noise: float = 0.4,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x1 = rng.normal(size=rows)
        x2 = rng.normal(size=rows)
        score = coef1 * x1 + coef2 * x2 + rng.normal(scale=noise, size=rows)
        y = (score > np.median(score)).astype(int)
        return pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    return _build


@pytest.fixture
def multiclass_frame():
    def _build(rows_per_class: int = 40, seed: int = 11, scale: float = 0.35) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        labels = ["alpha", "beta", "gamma"]
        frames: list[pd.DataFrame] = []
        for idx, label in enumerate(labels):
            center = float(idx) * 2.5
            x1 = rng.normal(loc=center, scale=scale, size=rows_per_class)
            x2 = rng.normal(loc=-center, scale=scale, size=rows_per_class)
            frames.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
        return pd.concat(frames, ignore_index=True)

    return _build


@pytest.fixture
def regression_frame():
    def _build(
        rows: int = 80,
        seed: int = 17,
        coef1: float = 3.0,
        coef2: float = -1.5,
        noise: float = 0.1,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x1 = rng.normal(size=rows)
        x2 = rng.normal(loc=2.0, scale=0.5, size=rows)
        y = coef1 * x1 + coef2 * x2 + rng.normal(scale=noise, size=rows)
        return pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    return _build


@pytest.fixture
def frontier_frame():
    def _build(rows: int = 120, seed: int = 314) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x1 = rng.uniform(-2.0, 2.0, size=rows)
        x2 = rng.normal(size=rows)
        base = 1.8 + 1.5 * x1 - 0.4 * x2
        noise = rng.normal(scale=0.25 + 0.3 * np.abs(x1), size=rows)
        tail = rng.exponential(scale=0.25, size=rows)
        y = base + noise + tail
        return pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    return _build


@pytest.fixture
def panel_frame():
    def _build(n_units: int = 80, seed: int = 101) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        units = np.arange(n_units)
        rows = []
        for unit in units:
            treated = int(unit % 2 == 0)
            x = float(rng.normal())
            y_pre = 10.0 + 0.8 * x + rng.normal(scale=0.4)
            y_post = y_pre + 0.5 + 1.2 * treated + rng.normal(scale=0.4)
            rows.append(
                {
                    "unit_id": int(unit),
                    "time": 0,
                    "post": 0,
                    "treatment": treated,
                    "x": x,
                    "outcome": y_pre,
                }
            )
            rows.append(
                {
                    "unit_id": int(unit),
                    "time": 1,
                    "post": 1,
                    "treatment": treated,
                    "x": x,
                    "outcome": y_post,
                }
            )
        return pd.DataFrame(rows)

    return _build


@pytest.fixture
def unbalanced_binary_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = 2000
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    target = np.zeros(rows, dtype=int)
    positives = max(1, int(rows * 0.001))
    target[:positives] = 1
    rng.shuffle(target)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


@pytest.fixture
def categorical_frame() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    rows = 120
    region = rng.choice(["north", "south", "east", "west"], size=rows)
    channel = rng.choice(["web", "store", "partner"], size=rows)
    x_num = rng.normal(size=rows)
    target = 3.2 * x_num + (region == "north").astype(float) - (channel == "store").astype(float)
    target += rng.normal(scale=0.15, size=rows)
    return pd.DataFrame({"region": region, "channel": channel, "x_num": x_num, "target": target})


@pytest.fixture
def timeseries_frame() -> pd.DataFrame:
    rng = np.random.default_rng(31415)
    rows = 180
    t = np.arange(rows)
    trend = 0.04 * t
    seasonal = np.sin(2 * np.pi * t / 24.0)
    noise = rng.normal(scale=0.12, size=rows)
    y = 10.0 + trend + 0.8 * seasonal + noise
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2023-01-01", periods=rows, freq="D"),
            "x_trend": trend,
            "x_seasonal": seasonal,
            "target": y,
        }
    )


@pytest.fixture
def missing_values_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2718)
    rows = 140
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "target": rng.normal(size=rows),
        }
    )
    for ratio, column in [(0.05, "x1"), (0.20, "x2"), (0.50, "x3")]:
        count = int(rows * ratio)
        idx = rng.choice(rows, size=count, replace=False)
        frame.loc[idx, column] = np.nan
    return frame


@pytest.fixture
def outlier_frame() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    rows = 120
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    target = 1.5 * x1 - 0.7 * x2 + rng.normal(scale=0.2, size=rows)
    outlier_idx = rng.choice(rows, size=5, replace=False)
    target[outlier_idx] += rng.normal(loc=25.0, scale=2.0, size=5)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


@pytest.fixture
def config_payload():
    def _build(task_type: str, **overrides: object) -> dict:
        base: dict = {
            "config_version": 1,
            "task": {"type": task_type},
            "data": {"path": "dummy.csv", "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 7},
            "export": {"artifact_dir": "artifacts"},
        }
        if task_type == "binary":
            base["split"] = {"type": "stratified", "n_splits": 3, "seed": 7}
            base["postprocess"] = {"calibration": "platt"}
        if task_type == "multiclass":
            base["split"] = {"type": "stratified", "n_splits": 3, "seed": 7}
        if task_type == "frontier":
            base["frontier"] = {"alpha": 0.90}
        return _deep_merge(base, overrides)

    return _build


@pytest.fixture
def FakeBooster():
    class _FakeBooster:
        def __init__(self, pred_value: float = 0.5) -> None:
            self.best_iteration = 1
            self._pred_value = pred_value

        def predict(self, x: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
            _ = num_iteration
            return np.full(len(x), self._pred_value, dtype=float)

        def model_to_string(self) -> str:
            return "fake-model"

    return _FakeBooster
