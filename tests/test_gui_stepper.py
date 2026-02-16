from __future__ import annotations

from veldra.gui import app as app_module


def test_stepper_with_completed_state() -> None:
    state = {
        "data_path": "x.csv",
        "target_col": "target",
        "task_type": "binary",
        "split_config": {"type": "stratified", "n_splits": 5, "seed": 42},
        "train_config": {"learning_rate": 0.05},
        "last_job_succeeded": True,
        "last_run_artifact": "artifacts/run1",
    }
    bar = app_module._stepper_bar("/results", state)
    assert bar.className == "stepper-container"
    text = str(bar)
    assert "Results" in text
    assert "✓" in text
