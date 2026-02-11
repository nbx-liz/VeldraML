from __future__ import annotations

from pathlib import Path

import pandas as pd

from examples.generate_data_dr import generate_dr_datasets


def test_generate_data_dr_writes_train_and_test_csv(tmp_path) -> None:
    stats = generate_dr_datasets(n_train=120, n_test=80, seed=101, out_dir=tmp_path)

    train_path = Path(stats["train_path"])
    test_path = Path(stats["test_path"])
    assert train_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert {"treatment", "outcome", "true_tau"} <= set(train_df.columns)
    assert {"treatment", "outcome", "true_tau"} <= set(test_df.columns)
    assert train_df["treatment"].isin([0, 1]).all()
    assert test_df["treatment"].isin([0, 1]).all()
