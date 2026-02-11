from __future__ import annotations

import json
import sys

import pandas as pd

from examples.generate_data_drdid import main


def test_generate_data_drdid_script_outputs_expected_files(tmp_path) -> None:
    out_dir = tmp_path / "data"
    argv = sys.argv
    try:
        sys.argv = [
            "generate_data_drdid.py",
            "--n-units",
            "12",
            "--n-pre",
            "20",
            "--n-post",
            "22",
            "--seed",
            "9",
            "--out-dir",
            str(out_dir),
        ]
        main()
    finally:
        sys.argv = argv

    panel_path = out_dir / "drdid_panel.csv"
    repeated_path = out_dir / "drdid_repeated_cs.csv"
    summary_path = out_dir / "drdid_summary.json"
    assert panel_path.exists()
    assert repeated_path.exists()
    assert summary_path.exists()

    panel = pd.read_csv(panel_path)
    repeated = pd.read_csv(repeated_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {"unit_id", "time", "post", "treatment", "outcome"} <= set(panel.columns)
    assert {"time", "post", "treatment", "outcome"} <= set(repeated.columns)
    assert summary["panel_rows"] == len(panel)
    assert summary["repeated_cs_rows"] == len(repeated)
