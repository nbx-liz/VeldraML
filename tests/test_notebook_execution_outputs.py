from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.notebook_e2e

SUMMARY_FILES = {
    "UC-1": Path("examples/out/phase35_uc01_regression_fit_evaluate/summary.json"),
    "UC-2": Path("examples/out/phase35_uc02_binary_fit_evaluate/summary.json"),
    "UC-3": Path("examples/out/phase35_uc03_multiclass_fit_evaluate/summary.json"),
    "UC-4": Path("examples/out/phase35_uc04_timeseries_fit_evaluate/summary.json"),
    "UC-5": Path("examples/out/phase35_uc05_frontier_fit_evaluate/summary.json"),
    "UC-6": Path("examples/out/phase35_uc06_dr_estimate/summary.json"),
    "UC-7": Path("examples/out/phase35_uc07_drdid_estimate/summary.json"),
    "UC-8": Path("examples/out/phase35_uc08_artifact_evaluate/summary.json"),
    "UC-9": Path("examples/out/phase35_uc09_binary_tune_evaluate/summary.json"),
    "UC-10": Path("examples/out/phase35_uc10_timeseries_tune_evaluate/summary.json"),
    "UC-11": Path("examples/out/phase35_uc11_frontier_tune_evaluate/summary.json"),
    "UC-12": Path("examples/out/phase35_uc12_dr_tune_estimate/summary.json"),
    "UC-13": Path("examples/out/phase35_uc13_drdid_tune_estimate/summary.json"),
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_quick_reference_outputs_have_materialized_files() -> None:
    for uc, summary_path in SUMMARY_FILES.items():
        assert summary_path.exists(), uc
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload.get("uc") == uc
        assert payload.get("status") == "passed", uc
        outputs = payload.get("outputs")
        assert isinstance(outputs, list) and outputs, uc

        csv_seen = 0
        png_seen = 0
        for out in outputs:
            path = Path(str(out))
            assert path.exists(), path
            assert path.stat().st_size > 0
            if path.suffix.lower() == ".png":
                png_seen += 1
                assert path.read_bytes()[:8] == PNG_SIGNATURE, path
            if path.suffix == ".csv":
                csv_seen += 1
                frame = pd.read_csv(path)
                assert not frame.empty, path

        assert csv_seen > 0, uc
        assert png_seen > 0, uc
