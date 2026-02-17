from __future__ import annotations

from veldra.causal.dr import run_dr_estimation
from veldra.config.models import RunConfig


def test_dr_output_includes_nuisance_diagnostics(tmp_path, regression_frame) -> None:
    frame = regression_frame(rows=60)
    frame["treatment"] = (frame["x1"] > float(frame["x1"].median())).astype(int)

    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": str(tmp_path / "dummy.csv"), "target": "target"},
            "split": {"type": "kfold", "n_splits": 3, "seed": 1},
            "causal": {"method": "dr", "treatment_col": "treatment"},
        }
    )
    out = run_dr_estimation(cfg, frame)
    assert "propensity_importance" in out.nuisance_diagnostics
    assert "outcome_importance" in out.nuisance_diagnostics
    assert "oof_metrics" in out.nuisance_diagnostics
