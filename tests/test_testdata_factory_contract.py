from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_PATTERN = re.compile(
    r"def (_binary_frame|_multiclass_frame|_regression_frame|_frontier_frame|_panel_frame)\s*\("
)

WAVE1_FILES = [
    "tests/test_binary_fit_smoke.py",
    "tests/test_binary_predict_contract.py",
    "tests/test_binary_evaluate_metrics.py",
    "tests/test_binary_artifact_roundtrip.py",
    "tests/test_multiclass_fit_smoke.py",
    "tests/test_multiclass_predict_contract.py",
    "tests/test_multiclass_evaluate_metrics.py",
    "tests/test_multiclass_artifact_roundtrip.py",
    "tests/test_frontier_fit_smoke.py",
    "tests/test_frontier_predict_contract.py",
    "tests/test_frontier_evaluate_metrics.py",
    "tests/test_frontier_artifact_roundtrip.py",
    "tests/test_binary_internal.py",
    "tests/test_multiclass_internal.py",
    "tests/test_regression_internal.py",
    "tests/test_frontier_internal.py",
    "tests/test_drdid_internal.py",
]


def test_wave1_files_do_not_define_local_testdata_frames() -> None:
    violations: list[str] = []
    for rel_path in WAVE1_FILES:
        text = Path(rel_path).read_text(encoding="utf-8")
        if FORBIDDEN_PATTERN.search(text):
            violations.append(rel_path)
    assert not violations, f"Wave1 files still define local test-data helpers: {violations}"
