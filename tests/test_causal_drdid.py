from __future__ import annotations

from veldra.causal.dr_did import run_dr_did_estimation
from veldra.config.models import RunConfig


def test_drdid_output_includes_parallel_trends_diagnostics(panel_frame) -> None:
    frame = panel_frame(n_units=40)
    frame = frame.rename(columns={"outcome": "target"})

    cfg = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"path": "dummy.csv", "target": "target", "id_cols": ["unit_id"]},
            "split": {"type": "group", "n_splits": 2, "group_col": "unit_id", "seed": 1},
            "causal": {
                "method": "dr_did",
                "treatment_col": "treatment",
                "design": "panel",
                "time_col": "time",
                "post_col": "post",
                "unit_id_col": "unit_id",
                "estimand": "att",
            },
        }
    )
    out = run_dr_did_estimation(cfg, frame)
    assert "parallel_trends" in out.summary
    assert "parallel_trends" in out.nuisance_diagnostics
