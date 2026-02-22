from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.generate_frontier_demo_data import OUTPUT_COLS, generate_frontier_demo_data


def _read_outputs(out_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = out_dir / "train_eval.csv"
    latest_path = out_dir / "latest.csv"
    assert train_path.exists()
    assert latest_path.exists()
    return pd.read_csv(train_path), pd.read_csv(latest_path)


def test_generate_frontier_demo_data_contract(tmp_path) -> None:
    out_dir = tmp_path / "frontier"
    train_df, latest_df = generate_frontier_demo_data(out_dir=out_dir, seed=42)

    assert len(train_df) == 12000
    assert len(latest_df) == 6000
    assert list(train_df.columns) == OUTPUT_COLS
    assert list(latest_df.columns) == OUTPUT_COLS

    persisted_train, persisted_latest = _read_outputs(out_dir)
    assert list(persisted_train.columns) == OUTPUT_COLS
    assert list(persisted_latest.columns) == OUTPUT_COLS
    assert len(persisted_train) == 12000
    assert len(persisted_latest) == 6000

    assert persisted_train["month"].min() == "2023-01"
    assert persisted_train["month"].max() == "2024-12"
    assert persisted_latest["month"].min() == "2025-01"
    assert persisted_latest["month"].max() == "2025-12"
    assert persisted_train["month"].nunique() == 24
    assert persisted_latest["month"].nunique() == 12

    assert (persisted_train["net_sales"] > 0).mean() > 0.5
    assert (persisted_latest["net_sales"] > 0).mean() > 0.5

    sec_level = persisted_train[["sec_id", "u_s"]].drop_duplicates()
    assert sec_level["u_s"].var() > 0.0

    median_by_sec = persisted_train.groupby("sec_id")["net_sales"].median()
    u_by_sec = sec_level.set_index("sec_id")["u_s"]
    q10 = u_by_sec.quantile(0.10)
    q90 = u_by_sec.quantile(0.90)
    low_u_sales = median_by_sec[u_by_sec <= q10].median()
    high_u_sales = median_by_sec[u_by_sec >= q90].median()
    assert low_u_sales > high_u_sales


def test_generate_frontier_demo_data_is_reproducible(tmp_path) -> None:
    out_dir_1 = tmp_path / "run1"
    out_dir_2 = tmp_path / "run2"
    generate_frontier_demo_data(out_dir=out_dir_1, seed=42)
    generate_frontier_demo_data(out_dir=out_dir_2, seed=42)

    train_1, latest_1 = _read_outputs(out_dir_1)
    train_2, latest_2 = _read_outputs(out_dir_2)
    assert_frame_equal(train_1, train_2)
    assert_frame_equal(latest_1, latest_2)
