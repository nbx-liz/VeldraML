from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.notebook_e2e

SUMMARY_FILES = {
    "UC-1": Path("examples/out/phase26_2_uc01_regression_fit_evaluate/summary.json"),
    "UC-2": Path("examples/out/phase26_2_uc02_binary_tune_evaluate/summary.json"),
    "UC-3": Path("examples/out/phase26_2_uc03_frontier_fit_evaluate/summary.json"),
    "UC-4": Path("examples/out/phase26_2_uc04_causal_dr_estimate/summary.json"),
    "UC-5": Path("examples/out/phase26_2_uc05_causal_drdid_estimate/summary.json"),
    "UC-6": Path("examples/out/phase26_2_uc06_causal_dr_tune/summary.json"),
    "UC-7": Path("examples/out/phase26_2_uc07_artifact_evaluate/summary.json"),
    "UC-8": Path("examples/out/phase26_2_uc08_artifact_reevaluate/summary.json"),
    "UC-11": Path("examples/out/phase26_3_uc_multiclass_fit_evaluate/summary.json"),
    "UC-12": Path("examples/out/phase26_3_uc_timeseries_fit_evaluate/summary.json"),
}


def test_phase26_3_summary_entries_and_outputs_exist() -> None:
    for uc, summary_path in SUMMARY_FILES.items():
        assert summary_path.exists(), uc
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload.get("uc") == uc
        assert payload.get("status") == "passed", uc
        assert payload.get("artifact_path"), uc
        assert payload.get("metrics"), uc
        outputs = payload.get("outputs", [])
        assert outputs, uc
        assert len(outputs) >= 3
        for out in outputs:
            assert Path(str(out)).exists(), out
