from __future__ import annotations

from veldra.gui.services import validate_config_with_guidance

_VALID_YAML = """
config_version: 1
task:
  type: regression
data:
  path: examples/data/causal_dr_tune_demo.csv
  target: target
split:
  type: kfold
  n_splits: 5
  seed: 42
train:
  num_boost_round: 100
  lgb_params:
    learning_rate: 0.05
    num_leaves: 31
    max_depth: -1
    min_child_samples: 20
    subsample: 1.0
    colsample_bytree: 1.0
    reg_alpha: 0.0
    reg_lambda: 0.0
  early_stopping_rounds: 20
  seed: 42
export:
  artifact_dir: artifacts
"""


_INVALID_YAML = "task:\n  type: regression\n"


def test_validate_config_with_guidance_ok() -> None:
    result = validate_config_with_guidance(_VALID_YAML)
    assert result["ok"] is True
    assert result["errors"] == []


def test_validate_config_with_guidance_error() -> None:
    result = validate_config_with_guidance(_INVALID_YAML)
    assert result["ok"] is False
    assert len(result["errors"]) >= 1
    assert "path" in result["errors"][0]
