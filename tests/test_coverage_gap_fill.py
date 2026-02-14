from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from veldra.api.artifact import Artifact
from veldra.artifact import exporter
from veldra.causal import dr_did
from veldra.causal.dr import DREstimationOutput
from veldra.config import models
from veldra.config.models import RunConfig


# --- models.py coverage ---
def test_resolve_tuning_objective_errors():
    # Line 393: Unsupported causal method
    with pytest.raises(ValueError, match="Unsupported causal method"):
        models.resolve_tuning_objective("regression", "obj", causal_method="magic_method")

    # Line 407: Unsupported task type
    with pytest.raises(ValueError, match="Unsupported task type"):
        models.resolve_tuning_objective("magic_task", "obj")


# --- dr_did.py coverage ---
def test_dr_did_weight_calculation(monkeypatch):
    # We need to reach lines 296-300.
    # This happens when run_dr_estimation returns metrics WITHOUT "smd_max_weighted".
    # And we need to test both 'ate' and 'att' (default uses att weights if not ate).

    # Mock observation table
    obs_df = pd.DataFrame(
        {
            "e_hat": [0.5, 0.5],
            "treatment": [1, 0],
        }
    )

    # Mock run_dr_estimation using real dataclass
    mock_dr_out = DREstimationOutput(
        method="dr_did",
        estimand="att",
        estimate=0.1,
        std_error=0.01,
        ci_lower=0.0,
        ci_upper=0.2,
        metrics={"overlap_metric": 0.5},  # No smd_max_weighted
        observation_table=obs_df,
        summary={},
    )

    monkeypatch.setattr(dr_did, "run_dr_estimation", lambda c, f: mock_dr_out)
    monkeypatch.setattr(dr_did, "_base_validation", lambda c, f: ("treatment", "target", "post"))
    monkeypatch.setattr(
        dr_did, "_panel_to_pseudo_frame", lambda c, f: (pd.DataFrame(), pd.DataFrame())
    )
    monkeypatch.setattr(
        dr_did, "_repeated_cs_to_pseudo_frame", lambda c, f: (pd.DataFrame(), pd.DataFrame())
    )
    monkeypatch.setattr(dr_did, "max_standardized_mean_difference", lambda c, t, weights=None: 0.1)

    # Use real Config to support model_copy
    config = RunConfig(
        config_version=1,
        task={"type": "regression"},
        data={"target": "target"},
        causal={
            "method": "dr_did",
            "treatment_col": "treatment",
            "time_col": "time",
            "post_col": "post",
            "unit_id_col": "unit",
            "design": "panel",
        },
    )

    df = pd.DataFrame(
        {"post": [0, 1], "time": [1, 1], "target": [0, 0], "treatment": [1, 0], "unit": [1, 1]}
    )

    # Run with ATT
    dr_did.run_dr_did_estimation(config, df)

    # Run with ATE (Line 297)
    config.causal.estimand = "ate"
    dr_did.run_dr_did_estimation(config, df)


# --- exporter.py coverage ---
def test_exporter_onnx_inference(monkeypatch, tmp_path):
    # Lines 428-444: check onnx runtime inference
    # We need to simulate a valid ONNX export verification where onnxruntime is present

    # Mock _validate_onnx_export internals
    # We assume export_onnx_model calls _validate_onnx_export if we test integration,
    # but simpler to test _validate_onnx_export directly if possible.
    # It is private.

    # We can invoke it via private access or test export_onnx_model if it calls it.
    # But export_onnx_model calls _validate_onnx_export implicitly?
    # No, usually export() calls validate().
    # Or export_onnx_model returns path, and we call validation manually?
    # Validating onnx export is usually done inside the pipeline or separate tool?
    # The code shown has `_validate_onnx_export` at module level (L350).
    # But checking calls... it is used?
    # It is NOT used in `export_onnx_model` (L125).
    # It returns path.
    # It seems `export_onnx_model` creates metadata but doesn't run `_validate_onnx_export`.
    # `validate_export` (not shown in snippet but likely exists) might call it.

    # Access it directly from module
    validate_func = exporter._validate_onnx_export

    # Setup artifact and directory
    artifact = MagicMock(spec=Artifact)
    artifact.feature_schema = {"feature_names": ["f1"]}

    # Setup files
    (tmp_path / "model.onnx").touch()
    (tmp_path / "metadata.json").write_text("{}")

    # Mock onnx, onnxruntime
    with patch.dict(
        "sys.modules",
        {"onnx": MagicMock(), "onnxruntime": MagicMock(), "onnx.checker": MagicMock()},
    ):
        # Mock InferenceSession
        mock_sess = MagicMock()
        mock_sess.get_inputs.return_value = [MagicMock(name="input")]
        mock_sess.run.return_value = [np.array([1.0])]  # Success logic

        sys.modules["onnxruntime"].InferenceSession.return_value = mock_sess

        # Test Success
        res = validate_func(tmp_path, artifact)
        assert res["validation_passed"] is True

        # Test Failure (Session run raises)
        mock_sess.run.side_effect = Exception("Inference failed")
        validate_func(tmp_path, artifact)

        # checks are in validation_report.json
        report_path = tmp_path / "validation_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text("utf-8"))
        checks = report["checks"]

        inf_check = next(c for c in checks if c["name"] == "onnx_runtime_inference")
        assert inf_check["ok"] is False
