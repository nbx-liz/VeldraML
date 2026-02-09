import pandas as pd
import pytest

from veldra.api import fit


def test_fit_reproducibility_with_fixed_seed(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "x1": [float(i) for i in range(50)],
            "x2": [float((i * 3) % 7) for i in range(50)],
            "y": [float(i * 1.7 + ((i * 3) % 7) * 0.5) for i in range(50)],
        }
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)

    payload = {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": str(path), "target": "y"},
        "split": {"type": "kfold", "n_splits": 5, "seed": 21},
        "train": {
            "seed": 21,
            "lgb_params": {
                "num_threads": 1,
                "deterministic": True,
                "force_col_wise": True,
            },
        },
        "export": {"artifact_dir": str(tmp_path / "artifacts")},
    }

    first = fit(payload)
    second = fit(payload)

    for key in ["rmse", "mae", "r2"]:
        assert key in first.metrics
        assert key in second.metrics
        assert first.metrics[key] == pytest.approx(second.metrics[key], abs=1e-12)
