# VeldraML

VeldraML is a config-driven LightGBM analysis library.
Current implemented tasks are regression, binary classification, multiclass classification, and frontier.

## Features

- Stable public API under `veldra.api.*`
- Config-driven execution via `RunConfig`
- Artifact-based reproducibility (`save/load/predict/evaluate`)
- Regression workflow: `fit`, `predict`, `evaluate`
- Binary workflow: `fit`, `predict`, `evaluate` with OOF-based Platt calibration
- Multiclass workflow: `fit`, `predict`, `evaluate`
- Frontier workflow: `fit`, `predict`, `evaluate` (quantile baseline)
- Hyperparameter tuning workflow: `tune` (regression/binary/multiclass/frontier)
- Scenario simulation workflow: `simulate` (regression/binary/multiclass/frontier)
- Export workflow: `export` (`python` + optional `onnx`)
- Causal workflow: `estimate_dr` (DR + DR-DiD, ATT default)

## Project Status

Implemented:
- `fit`, `predict`, `evaluate` for `task.type=regression`
- `fit`, `predict`, `evaluate` for `task.type=binary`
- `fit`, `predict`, `evaluate` for `task.type=multiclass`
- `fit`, `predict`, `evaluate` for `task.type=frontier`
- `tune` for `task.type=regression|binary|multiclass|frontier` (Optuna-based MVP)
- `simulate` for `task.type=regression|binary|multiclass|frontier` (Scenario DSL MVP)
- `export` for all implemented tasks (`python` always, `onnx` optional dependency)
- `estimate_dr` for:
  - `causal.method=dr` with `task.type=regression|binary` (ATT default, OOF-calibrated propensity)
  - `causal.method=dr_did` with `task.type=regression` (2-period panel/repeated cross-section)

Backlog:
- ONNX graph optimization pipeline (quantization is available as opt-in)
- Causal DiD extensions beyond 2-period MVP (multi-period / staggered adoption)

## Requirements

- Python `>=3.11`
- Dependency management: `uv`

## Installation

### Development install

```bash
uv sync --dev
```

Optional ONNX export dependencies:

```bash
uv sync --extra export-onnx
```

### Verify installation

```bash
uv run python -c "import veldra; print(veldra.__version__)"
```

## Quick Start

### API usage (regression)

```python
from veldra.api import Artifact, evaluate, fit, predict
from veldra.data import load_tabular_data

config = {
    "config_version": 1,
    "task": {"type": "regression"},
    "data": {"path": "train.csv", "target": "target"},
    "split": {"type": "kfold", "n_splits": 5, "seed": 42},
    "export": {"artifact_dir": "artifacts"},
}

run = fit(config)
artifact = Artifact.load(run.artifact_path)
frame = load_tabular_data("test.csv")
pred = predict(artifact, frame.drop(columns=["target"]))
ev = evaluate(artifact, frame)
# or evaluate directly from config (ephemeral train + evaluate, no artifact save):
ev_cfg = evaluate(config, frame)
```

### API usage (causal DR, ATT default)

```python
from veldra.api import estimate_dr

dr_result = estimate_dr(
    {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "dr_train.csv", "target": "outcome"},
        "causal": {
            "method": "dr",
            "treatment_col": "treatment",
            "estimand": "att",  # default
            "propensity_calibration": "platt",  # default
        },
        "export": {"artifact_dir": "artifacts"},
    }
)
print(dr_result.estimate, dr_result.metrics["dr"])
```

### API usage (causal DR-DiD, panel)

```python
from veldra.api import estimate_dr

drdid_result = estimate_dr(
    {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "drdid_panel.csv", "target": "outcome"},
        "causal": {
            "method": "dr_did",
            "treatment_col": "treatment",
            "design": "panel",
            "time_col": "time",
            "post_col": "post",
            "unit_id_col": "unit_id",
            "estimand": "att",  # default
            "propensity_calibration": "platt",  # default
        },
        "export": {"artifact_dir": "artifacts"},
    }
)
print(drdid_result.estimate, drdid_result.metrics["drdid"])
```

Advanced time-series split options (non-default, opt-in):

```python
config = {
    "config_version": 1,
    "task": {"type": "regression"},
    "data": {"path": "train.csv", "target": "target"},
    "split": {
        "type": "timeseries",
        "time_col": "timestamp",
        "n_splits": 5,
        "timeseries_mode": "blocked",   # expanding | blocked
        "train_size": 200,              # required for blocked
        "test_size": 50,
        "gap": 5,
        "embargo": 10,
    },
    "export": {"artifact_dir": "artifacts"},
}
```

### API usage (tuning)

```python
from veldra.api import tune

tune_result = tune(
    {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": "train.csv", "target": "target"},
        "split": {"type": "kfold", "n_splits": 5, "seed": 42},
        "tuning": {"enabled": True, "n_trials": 20, "preset": "fast"},
        "export": {"artifact_dir": "artifacts"},
    }
)
print(tune_result.best_score, tune_result.best_params)
```

### Example scripts

Regression (California Housing):

```bash
uv run python examples/prepare_demo_data.py
uv run python examples/run_demo_regression.py
uv run python examples/evaluate_demo_artifact.py --artifact-path <artifact_dir>
```

Binary (Breast Cancer):

```bash
uv run python examples/prepare_demo_data_binary.py
uv run python examples/run_demo_binary.py
uv run python examples/evaluate_demo_binary_artifact.py --artifact-path <artifact_dir>
```

Optional: enable binary threshold optimization explicitly:

```bash
uv run python examples/run_demo_binary.py --optimize-threshold
```

Multiclass (Iris):

```bash
uv run python examples/prepare_demo_data_multiclass.py
uv run python examples/run_demo_multiclass.py
uv run python examples/evaluate_demo_multiclass_artifact.py --artifact-path <artifact_dir>
```

Frontier (synthetic quantile demo):

```bash
uv run python examples/prepare_demo_data_frontier.py
uv run python examples/run_demo_frontier.py --alpha 0.90
uv run python examples/evaluate_demo_frontier_artifact.py --artifact-path <artifact_dir>
```

Example outputs are saved under `examples/out/<timestamp>/`:

- `run_result.json`
- `eval_result.json`
- `predictions_sample.csv`
- `used_config.yaml`
- `artifacts/<run_id>/...`

Tune demo (single script for all supported tasks):

```bash
uv run python examples/run_demo_tune.py --task regression --n-trials 10
uv run python examples/run_demo_tune.py --task binary --objective logloss --n-trials 20
uv run python examples/run_demo_tune.py --task multiclass --objective accuracy --n-trials 20
uv run python examples/run_demo_tune.py --task frontier --objective pinball --n-trials 20
# opt-in coverage-aware frontier objective
uv run python examples/run_demo_tune.py --task frontier --objective pinball_coverage_penalty --coverage-target 0.92 --coverage-tolerance 0.02 --penalty-weight 2.0 --n-trials 20
# causal DR tuning objective (nuisance models)
uv run python examples/run_demo_tune.py --task regression --objective dr_std_error --n-trials 20
# causal DR-DiD tuning objective
uv run python examples/run_demo_tune.py --task regression --objective drdid_std_error --n-trials 20
```

Tune resume / verbosity / custom search-space:

```bash
uv run python examples/run_demo_tune.py \
  --task regression \
  --study-name my_study \
  --resume \
  --log-level DEBUG \
  --search-space-file examples/search_space_regression.yaml
```

DR-DiD synthetic data generation:

```bash
uv run python examples/generate_data_drdid.py --n-units 3000 --n-pre 2500 --n-post 2500 --seed 42
```

Simulate demo (regression baseline with scenario actions):

```bash
uv run python examples/run_demo_simulate.py --data-path examples/data/california_housing.csv
```

Notebook sample (regression workflow with diagnostics):

```bash
# Open and run notebooks/regression_analysis_workflow.ipynb
```

Notebook includes:
- Train/Test prediction vs actual comparison (table + plots)
- Error analysis (histogram/boxplot)
- LightGBM feature importance (table + bar chart)
- SHAP-style contribution summary via LightGBM `pred_contrib=True` (table + bar chart)
- Simulation and export cells

Notebook sample (frontier workflow with diagnostics):

```bash
# Open and run notebooks/frontier_analysis_workflow.ipynb
```

Notebook includes:
- In-notebook synthetic frontier data generation (base/drift)
- Train/Test frontier prediction vs actual comparison
- `u_hat`, `coverage`, and `pinball` diagnostics
- LightGBM feature importance and pred_contrib-based SHAP summary
- Simulation and export cells

Notebook sample (simulate-focused what-if analysis):

```bash
# Open and run notebooks/simulate_analysis_workflow.ipynb
```

Notebook includes:
- In-notebook synthetic SaaS data generation
- Scenario design for intervention analysis
- Scenario KPI table (`mean_uplift`, `uplift_win_rate`, `downside_rate`)
- Segment-level uplift analysis
- Top impacted account shortlist for operational rollout

Notebook sample (binary + tune analysis):

```bash
# Open and run notebooks/binary_tuning_analysis_workflow.ipynb
```

Notebook includes:
- In-notebook synthetic binary data generation (base + drift)
- Baseline binary fit/evaluate with calibrated probability outputs
- Hyperparameter tuning (`tune`) and trial history visualization
- Baseline vs tuned metric comparison
- ROC / confusion matrix / error-distribution diagnostics

Notebook sample (Lalonde DR causal analysis):

```bash
# Open and run notebooks/lalonde_dr_analysis_workflow.ipynb
```

Notebook includes:
- Public URL ingestion for Lalonde data with local cache reuse
- DR causal estimation via `estimate_dr(config)` with explicit ATT/platt defaults
- Naive/IPW/DR comparison table and ATT estimate chart with confidence interval
- Propensity diagnostics (`e_raw` and `e_hat`) by treated/control
- Balance diagnostics via SMD (unweighted vs ATT-weighted)

Notebook sample (Lalonde DR-DiD causal analysis):

```bash
# Open and run notebooks/lalonde_drdid_analysis_workflow.ipynb
```

Notebook includes:
- Public URL ingestion for Lalonde data with local cache reuse
- Panel transformation using pre=`re75` and post=`re78` outcomes
- DR-DiD causal estimation via `estimate_dr(config)` with:
  - `causal.method="dr_did"`
  - `causal.design="panel"`
  - explicit ATT/platt defaults
- Naive/IPW/DR/DR-DiD comparison table with CI
- Propensity diagnostics (`e_raw` and `e_hat`) and overlap summary
- Balance diagnostics via SMD (unweighted vs ATT-weighted)

Export demo:

```bash
uv run python examples/run_demo_export.py --artifact-path <artifact_dir> --format python
```

Export writes a machine-readable validation report at:
- `<export_dir>/validation_report.json`
- `ExportResult.metadata` includes `validation_passed`, `validation_report`, and `validation_mode`.

ONNX export (requires optional dependencies):

```bash
uv sync --extra export-onnx
uv run python examples/run_demo_export.py --artifact-path <artifact_dir> --format onnx
# frontier artifact is also supported:
uv run python examples/run_demo_export.py --artifact-path <frontier_artifact_dir> --format onnx
```

ONNX dynamic quantization (opt-in, default off):

```yaml
export:
  artifact_dir: artifacts
  onnx_optimization:
    enabled: true
    mode: dynamic_quant
```

When enabled, export also writes `model.optimized.onnx` and records size diff metadata.

## Development

Run quality checks:

```bash
uv run ruff check .
uv run pytest -q
```

Coverage:

```bash
uv run coverage run -m pytest -q
uv run coverage report -m
```

## Project Structure

```text
src/veldra/
  api/          # stable public API
  config/       # RunConfig models and IO
  modeling/     # task-specific training logic
  artifact/     # artifact persistence
  data/         # data loaders
  split/        # split strategies
  simulate/     # scenario simulation engine
examples/       # runnable demos
tests/          # unit tests
```

## Notes

- Root `README.md` is the single source of truth for project documentation.
- Design decisions and phase logs are tracked in `DESIGN_BLUEPRINT.md` and `HISTORY.md`.

