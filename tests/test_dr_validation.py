from __future__ import annotations

import pandas as pd
import pytest

from veldra.api import VeldraNotImplementedError, VeldraValidationError, estimate_dr


def test_estimate_dr_requires_causal_config(tmp_path) -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "treatment": [0, 1, 0, 1], "y": [1.0, 2.0, 1.2, 2.4]}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError):
        estimate_dr(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "y"},
                "export": {"artifact_dir": str(tmp_path)},
            }
        )


def test_estimate_dr_rejects_unsupported_task(tmp_path) -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "treatment": [0, 1, 0, 1], "y": ["a", "b", "a", "b"]}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraNotImplementedError):
        estimate_dr(
            {
                "config_version": 1,
                "task": {"type": "multiclass"},
                "data": {"path": str(path), "target": "y"},
                "causal": {"treatment_col": "treatment"},
                "export": {"artifact_dir": str(tmp_path)},
            }
        )


def test_estimate_dr_rejects_non_binary_treatment(tmp_path) -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "treatment": [0, 1, 2, 1], "y": [1.0, 2.0, 1.1, 2.2]}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(VeldraValidationError):
        estimate_dr(
            {
                "config_version": 1,
                "task": {"type": "regression"},
                "data": {"path": str(path), "target": "y"},
                "causal": {"treatment_col": "treatment"},
                "export": {"artifact_dir": str(tmp_path)},
            }
        )
