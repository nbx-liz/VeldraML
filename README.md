# VeldraML

VeldraML is a config-driven LightGBM analysis library.
Current implemented tasks are regression, binary classification, multiclass classification, and frontier.

## Why VeldraML?

Teams often hit the same operational problems:
- experimentation and production paths drift into separate interfaces
- flexible configs become inconsistent and reduce reproducibility
- training outputs are not packaged as a portable handoff unit

VeldraML addresses those gaps with explicit contracts:
- one entrypoint contract: `RunConfig`
- one handoff contract: `Artifact`
- one stable adapter surface: `veldra.api.*`

When to use:
- config-driven ML analysis workflows where reproducibility and traceability matter.

When not to use:
- a distributed training platform or online serving infrastructure by itself.

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
- GUI workflow (optional): Dash enhanced MVP (`config`/`run`/`artifacts`)
- Config migration utility: `veldra config migrate` (v1 normalization)

## API Reference (Summary)

| API | Input | Output | Typical Raises |
| --- | --- | --- | --- |
| `fit(config)` | `RunConfig \| dict` | `RunResult` | `VeldraValidationError`, `VeldraNotImplementedError` |
| `tune(config)` | `RunConfig \| dict` | `TuneResult` | `VeldraValidationError` |
| `estimate_dr(config)` | `RunConfig \| dict` | `CausalResult` | `VeldraValidationError`, `VeldraNotImplementedError` |
| `evaluate(artifact_or_config, data)` | `Artifact \| RunConfig \| dict`, `pd.DataFrame` | `EvalResult` | `VeldraValidationError`, `VeldraNotImplementedError` |
| `predict(artifact, data)` | `Artifact`, `pd.DataFrame` | `Prediction` | `VeldraValidationError`, `VeldraNotImplementedError` |
| `simulate(artifact, data, scenarios)` | `Artifact`, `pd.DataFrame`, `dict \| list[dict]` | `SimulationResult` | `VeldraValidationError`, `VeldraNotImplementedError` |
| `export(artifact, format)` | `Artifact`, `"python" \| "onnx"` | `ExportResult` | `VeldraValidationError`, `VeldraNotImplementedError` |

## Algorithm Overview

- CV training (`fit`):
  - Uses split strategy from `RunConfig.split` (`kfold/group/stratified/timeseries`) and aggregates OOF metrics.
- Binary probability calibration:
  - Fits calibrator from OOF raw probabilities only to prevent leakage.
- DR estimation (`causal.method="dr"`):
  - Combines calibrated propensity and outcome nuisance models through doubly robust score equations.
- DR-DiD estimation (`causal.method="dr_did"`):
  - Converts data to pseudo outcomes (panel/repeated-cross-section), then applies DR pipeline with overlap/SMD diagnostics.
- Frontier training:
  - Optimizes quantile objective and reports pinball/coverage/inefficiency metrics.
- Scenario DSL simulation:
  - Applies validated feature perturbations (`set/add/mul/clip`) and reports baseline vs scenario deltas.

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
  - `causal.method=dr_did` with `task.type=regression|binary` (2-period panel/repeated cross-section)
- Dash GUI adapter MVP:
  - Guided workflow: `Data -> Target -> Validation -> Train -> Run -> Results`
  - Runs history + Compare view
  - Run console async queue (`fit/evaluate/tune/simulate/export/estimate_dr`)
  - Results explorer with learning curves and config view
  - Async report export (`export_excel`, `export_html_report`)
  - `/config` route is retained as compatibility entrypoint and redirects users to new flow
- Config migration utility:
  - `veldra config migrate --input <path> [--output <path>]`
  - strict validation + non-destructive output (`*.migrated.yaml`)
  - current scope supports only `config_version=1 -> target_version=1`

Backlog:
- Causal DiD extensions beyond 2-period MVP (multi-period / staggered adoption)
- Advanced simulation DSL operators (`allocate_total`, constrained allocation)
- Binary threshold optimization beyond F1-only objective
- Python export packaging enhancements (e.g. Dockerfile generation)

Status note:
- ONNX graph optimization is intentionally skipped (non-priority). Optional ONNX dynamic quantization remains supported.

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

Optional GUI dependencies:

```bash
uv sync --extra gui
```

Optional report-export extra:

```bash
uv sync --extra export-report
```

Note: `export-report` is a placeholder extra for environment-specific report dependencies
(e.g. SHAP). Install those packages manually when your NumPy/runtime combination is compatible.

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

### API usage (causal DR-DiD, binary outcome)

```python
from veldra.api import estimate_dr

drdid_binary_result = estimate_dr(
    {
        "config_version": 1,
        "task": {"type": "binary"},
        "data": {"path": "drdid_panel_binary.csv", "target": "outcome"},
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
print(drdid_binary_result.estimate, drdid_binary_result.metadata["outcome_scale"])
```

Binary DR-DiD estimates are interpreted as **Risk Difference ATT** (treated-group probability difference).

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

## From Quick Start to Production

Bridge to production by keeping one run contract and validating each stage.

### Step 1. Fix the data contract

- Inputs:
  - source dataset path
  - agreed columns (`target`, optional `treatment`, optional id columns)
- Command example:
```bash
uv run python -c "from veldra.data import load_tabular_data; df=load_tabular_data('train.csv'); print(df.shape, list(df.columns)[:10])"
```
- Success criteria:
  - required columns exist
  - target/treatment semantics are fixed for this run

### Step 2. Create RunConfig from template

- Inputs:
  - task type and split strategy
  - artifact root path
- Command example:
```bash
cp configs/gui_run.yaml configs/run.prod.yaml
uv run veldra config migrate --input configs/run.prod.yaml
```
- Success criteria:
  - config validates and migration reports expected version
  - output policy remains non-destructive

### Step 3. Execute fit/evaluate/tune in order

- Inputs:
  - validated RunConfig
  - train/eval datasets
- Command example:
```bash
uv run python examples/run_demo_regression.py
uv run python examples/evaluate_demo_artifact.py --artifact-path <artifact_dir>
uv run python examples/run_demo_tune.py --task regression --n-trials 20
```
- Success criteria:
  - artifact directory is created
  - metrics and tuning artifacts are persisted

### Step 4. Export and verify artifacts

- Inputs:
  - artifact path
  - export format
- Command example:
```bash
uv run python examples/run_demo_export.py --artifact-path <artifact_dir> --format python
```
- Success criteria:
  - export directory exists
  - `validation_report.json` is generated and reflected in metadata

### Step 5. Reproduce with the same contract

- Inputs:
  - same RunConfig, seeds, split settings
- Command example:
```bash
uv run python examples/run_demo_regression.py
```
- Success criteria:
  - outputs are reproducible within expected numeric tolerance
  - config and artifact metadata are traceable

### Step 6. GUI operations (optional)

- Inputs:
  - GUI runtime env vars and SQLite path
- Command example:
```bash
VELDRA_GUI_JOB_DB_PATH=.veldra_gui/jobs.sqlite3 uv run veldra-gui --host 127.0.0.1 --port 8050
```
- Success criteria:
  - async queue state is visible
  - queued cancel works immediately
  - running cancel is recorded as best-effort (`cancel_requested`)

### Production minimum checklist

- `config_version` is pinned and migration output is archived.
- `export.artifact_dir` is fixed per environment.
- export validation report is archived with run outputs.
- migration stays non-destructive (no overwrite).

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
# causal DR tuning (default objective: dr_balance_priority)
uv run python examples/run_demo_tune.py --task regression --objective dr_balance_priority --n-trials 20
# causal DR-DiD tuning (default objective: drdid_balance_priority)
uv run python examples/run_demo_tune.py --task regression --objective drdid_balance_priority --causal-method dr_did --causal-design panel --time-col time --post-col post --unit-id-col unit_id --n-trials 20
# legacy objectives remain available
uv run python examples/run_demo_tune.py --task regression --objective dr_std_error --n-trials 20
uv run python examples/run_demo_tune.py --task regression --objective drdid_std_error --causal-method dr_did --causal-design panel --time-col time --post-col post --unit-id-col unit_id --n-trials 20
# balance-priority threshold / penalty tuning knobs
uv run python examples/run_demo_tune.py --task regression --objective dr_balance_priority --causal-balance-threshold 0.08 --causal-penalty-weight 3.0 --n-trials 20
```

Note:
- If `--objective` is causal (`dr_*` / `drdid_*`) and `--data-path` is omitted, the demo script auto-generates a causal dataset under `examples/data/`.
- For non-resume runs without `--study-name`, the script assigns a per-run study name to avoid "study already exists" collisions.

Causal balance-priority objectives optimize in two stages:
1. satisfy `smd_max_weighted <= causal_balance_threshold`
2. minimize `std_error` among balanced candidates

If the threshold is violated, objective value is dominated by a large violation penalty.

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
# Open and run notebooks/tutorials/tutorial_01_regression_basics.ipynb
```

Notebook index (tutorials + quick references):

```bash
# Open and run notebooks/reference_index.ipynb
```

Notebook includes:
- Train/Test prediction vs actual comparison (table + plots)
- Error analysis (histogram/boxplot)
- LightGBM feature importance (table + bar chart)
- SHAP-style contribution summary via LightGBM `pred_contrib=True` (table + bar chart)
- Simulation and export cells

Notebook sample (frontier workflow with diagnostics):

```bash
# Open and run notebooks/tutorials/tutorial_03_frontier_quantile_regression.ipynb
```

Notebook includes:
- In-notebook synthetic frontier data generation (base/drift)
- Train/Test frontier prediction vs actual comparison
- `u_hat`, `coverage`, and `pinball` diagnostics
- LightGBM feature importance and pred_contrib-based SHAP summary
- Simulation and export cells

Notebook sample (simulate-focused what-if analysis):

```bash
# Open and run notebooks/tutorials/tutorial_04_scenario_simulation.ipynb
```

Notebook includes:
- In-notebook synthetic SaaS data generation
- Scenario design for intervention analysis
- Scenario KPI table (`mean_uplift`, `uplift_win_rate`, `downside_rate`)
- Segment-level uplift analysis
- Top impacted account shortlist for operational rollout

Notebook sample (binary + tune analysis):

```bash
# Open and run notebooks/tutorials/tutorial_02_binary_classification_tuning.ipynb
```

Notebook includes:
- In-notebook synthetic binary data generation (base + drift)
- Baseline binary fit/evaluate with calibrated probability outputs
- Hyperparameter tuning (`tune`) and trial history visualization
- Baseline vs tuned metric comparison
- ROC / confusion matrix / error-distribution diagnostics

Notebook sample (Lalonde DR causal analysis):

```bash
# Open and run notebooks/tutorials/tutorial_05_causal_dr_lalonde.ipynb
```

Notebook includes:
- Public URL ingestion for Lalonde data with local cache reuse
- DR causal estimation via `estimate_dr(config)` with explicit ATT/platt defaults
- Naive/IPW/DR comparison table and ATT estimate chart with confidence interval
- Propensity diagnostics (`e_raw` and `e_hat`) by treated/control
- Balance diagnostics via SMD (unweighted vs ATT-weighted)

Notebook sample (Lalonde DR-DiD causal analysis):

```bash
# Open and run notebooks/tutorials/tutorial_06_causal_drdid_lalonde.ipynb
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

Phase26.3 notebook execution policy:
- `notebooks/quick_reference/reference_01_regression_fit_evaluate.ipynb` 〜 `notebooks/quick_reference/reference_08_artifact_reevaluate.ipynb` と `notebooks/quick_reference/reference_11_multiclass_fit_evaluate.ipynb` / `notebooks/quick_reference/reference_12_timeseries_fit_evaluate.ipynb` は、実行済みセルをコミットして配布します。
- 実行証跡は `examples/out/phase26_*/summary.json` と生成物ファイル群で管理します。
- 構造契約は通常テストで検証し、重い証跡検証は `pytest -m notebook_e2e` で実行します。

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

GUI launch:

```bash
uv run veldra-gui --host 127.0.0.1 --port 8050
```

Linux/WSL quick start (launch server + open browser):

```bash
./scripts/start_gui.sh
```

Runtime environment options:

```bash
# SQLite persistence path for async GUI jobs
VELDRA_GUI_JOB_DB_PATH=.veldra_gui/jobs.sqlite3
# Run page polling interval (milliseconds)
VELDRA_GUI_POLL_MS=2000
```

GUI run behavior:
- `/run` enqueues jobs asynchronously and keeps history in SQLite.
- queued jobs can be canceled immediately.
- running jobs support best-effort cancellation (`cancel_requested`) and may still complete.
- Main GUI flow is `/data`, `/target`, `/validation`, `/train`, `/run`, `/results`.
- `/runs` provides history, clone, compare, and migrate actions.
- `/compare` shows metric/config diffs for two artifacts.
- `/config` remains available for compatibility and forwards users to `/target`.
- default run config path is `configs/gui_run.yaml` (auto-created if missing).

Windows quick start (launch server + open browser):

```powershell
scripts\start_gui.ps1
```

or

```bat
scripts\start_gui.cmd
```

## RunConfig Reference (Complete)

<!-- RUNCONFIG_REF:START -->
### Field Reference

_This section is auto-generated from `src/veldra/config/models.py`. Do not edit manually._

| path | type | required | default | allowed values | scope | description |
| --- | --- | --- | --- | --- | --- | --- |
| `config_version` | `int` | yes | - | - | - | Configuration schema version. |
| `task` | `TaskConfig` | yes | - | - | - | Task-level settings container. |
| `task.type` | ``regression` | `binary` | `multiclass` | `frontier`` | yes | - | `regression`, `binary`, `multiclass`, `frontier` | - | Task kind to execute. |
| `data` | `DataConfig` | yes | - | - | - | Input data contract settings. |
| `data.path` | `str | None` | no | `None` | - | - | Training data path (required for fit/tune/estimate_dr). |
| `data.target` | `str` | yes | - | - | - | Target column name. |
| `data.id_cols` | `list[str]` | no | `[]` | - | - | Identifier columns protected in simulation. |
| `data.categorical` | `list[str]` | no | `[]` | - | - | Categorical feature columns for LightGBM. |
| `data.drop_cols` | `list[str]` | no | `[]` | - | - | Columns dropped before training. |
| `split` | `SplitConfig` | no | `<factory>` | - | - | Cross-validation split settings. |
| `split.type` | ``kfold` | `stratified` | `group` | `timeseries`` | no | `kfold` | `kfold`, `stratified`, `group`, `timeseries` | - | Split strategy selector. |
| `split.n_splits` | `int` | no | `5` | - | - | Number of CV folds. |
| `split.time_col` | `str | None` | no | `None` | - | - | Time column for timeseries split. |
| `split.group_col` | `str | None` | no | `None` | - | - | Group column for group split. |
| `split.seed` | `int` | no | `42` | - | - | Random seed for split shuffling. |
| `split.timeseries_mode` | ``expanding` | `blocked`` | no | `expanding` | `expanding`, `blocked` | - | Timeseries CV mode. |
| `split.test_size` | `int | None` | no | `None` | - | - | Validation window size for timeseries split. |
| `split.gap` | `int` | no | `0` | - | - | Gap between train and validation windows. |
| `split.embargo` | `int` | no | `0` | - | - | Embargo window after validation horizon. |
| `split.train_size` | `int | None` | no | `None` | - | - | Fixed train window size in blocked mode. |
| `train` | `TrainConfig` | no | `<factory>` | - | - | Model training settings. |
| `train.lgb_params` | `dict[str, Any]` | no | `{}` | - | - | LightGBM parameter overrides. |
| `train.metrics` | `list[str] | None` | no | `None` | - | - | Training metric list passed to LightGBM. |
| `train.early_stopping_rounds` | `int | None` | no | `100` | - | - | Early stopping rounds. |
| `train.early_stopping_validation_fraction` | `float` | no | `0.1` | - | - | Train-row fraction used for ES validation split. |
| `train.num_boost_round` | `int` | no | `300` | - | - | Maximum boosting iterations. |
| `train.auto_class_weight` | `bool` | no | `true` | - | task.type in {binary,multiclass} | Auto class balancing for binary/multiclass tasks. |
| `train.class_weight` | `dict[str, float] | None` | no | `None` | - | task.type in {binary,multiclass} | Manual class weights by label. |
| `train.auto_num_leaves` | `bool` | no | `false` | - | - | Auto-resolve num_leaves from max_depth. |
| `train.num_leaves_ratio` | `float` | no | `1.0` | - | - | Ratio applied to auto-resolved num_leaves. |
| `train.min_data_in_leaf_ratio` | `float | None` | no | `None` | - | - | Ratio-based min_data_in_leaf override. |
| `train.min_data_in_bin_ratio` | `float | None` | no | `None` | - | - | Ratio-based min_data_in_bin override. |
| `train.feature_weights` | `dict[str, float] | None` | no | `None` | - | - | Feature weights map keyed by feature name. |
| `train.top_k` | `int | None` | no | `None` | - | task.type=binary | Precision@k setting for binary task. |
| `train.seed` | `int` | no | `42` | - | - | Training seed. |
| `tuning` | `TuningConfig` | no | `<factory>` | - | - | Hyperparameter tuning settings. |
| `tuning.enabled` | `bool` | no | `false` | - | - | Enable/disable tuning path. |
| `tuning.n_trials` | `int` | no | `30` | - | - | Optuna trial count. |
| `tuning.search_space` | `dict[str, Any]` | no | `{}` | - | - | Explicit search space spec. |
| `tuning.metrics_candidates` | `list[str] | None` | no | `None` | - | - | Candidate metrics list for tuning diagnostics/reporting. |
| `tuning.preset` | ``fast` | `standard`` | no | `standard` | `fast`, `standard` | - | Default search space preset. |
| `tuning.objective` | `str | None` | no | `None` | - | - | Objective metric name. |
| `tuning.resume` | `bool` | no | `false` | - | - | Resume an existing study. |
| `tuning.study_name` | `str | None` | no | `None` | - | - | Explicit study name. |
| `tuning.log_level` | ``DEBUG` | `INFO` | `WARNING` | `ERROR`` | no | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | - | Tuning log verbosity. |
| `tuning.coverage_target` | `float | None` | no | `None` | - | task.type=frontier | Target coverage for frontier tuning. |
| `tuning.coverage_tolerance` | `float` | no | `0.01` | - | task.type=frontier | Allowed coverage deviation. |
| `tuning.penalty_weight` | `float` | no | `1.0` | - | task.type=frontier | Coverage penalty weight. |
| `tuning.causal_penalty_weight` | `float` | no | `1.0` | - | causal configured | Penalty weight for causal objectives. |
| `tuning.causal_balance_threshold` | `float` | no | `0.1` | - | causal configured | Max weighted SMD threshold. |
| `postprocess` | `PostprocessConfig` | no | `<factory>` | - | - | Postprocessing settings. |
| `postprocess.calibration` | ``platt` | `isotonic` | None` | no | `None` | `platt`, `isotonic` | task.type=binary | Probability calibration method. |
| `postprocess.threshold` | `float | None` | no | `None` | - | task.type=binary | Fixed decision threshold for binary. |
| `postprocess.threshold_optimization` | `ForwardRef("'ThresholdOptimizationConfig | None'")` | no | `None` | - | task.type=binary | Threshold optimization settings. |
| `simulation` | `SimulationConfig` | no | `<factory>` | - | - | Simulation DSL settings. |
| `simulation.scenarios` | `list[dict[str, Any]]` | no | `[]` | - | - | Scenario list. |
| `simulation.actions` | `list[dict[str, Any]]` | no | `[]` | - | - | Action list (legacy/helper). |
| `simulation.constraints` | `list[dict[str, Any]]` | no | `[]` | - | - | Constraint list (reserved). |
| `export` | `ExportConfig` | no | `<factory>` | - | - | Artifact/export settings. |
| `export.artifact_dir` | `str` | no | `artifacts` | - | - | Artifact root directory. |
| `export.inference_package` | `bool` | no | `false` | - | - | Reserved package export toggle. |
| `export.onnx_optimization` | `ForwardRef("'OnnxOptimizationConfig'")` | no | `<factory>` | - | - | ONNX optimization settings. |
| `frontier` | `FrontierConfig` | no | `<factory>` | - | task.type=frontier | Frontier task settings. |
| `frontier.alpha` | `float` | no | `0.9` | - | task.type=frontier | Quantile alpha (0,1). |
| `causal` | `CausalConfig | None` | no | `None` | - | task.type in {regression,binary} | Causal estimation settings. |
| `causal.method` | ``dr` | `dr_did`` | no | `dr` | `dr`, `dr_did` | - | Causal estimator family. |
| `causal.treatment_col` | `str` | yes | - | - | - | Treatment column name. |
| `causal.estimand` | ``att` | `ate`` | no | `att` | `att`, `ate` | - | Target estimand. |
| `causal.design` | ``panel` | `repeated_cross_section` | None` | no | `None` | `panel`, `repeated_cross_section` | causal.method=dr_did | DR-DiD design type. |
| `causal.time_col` | `str | None` | no | `None` | - | causal.method=dr_did | Time column for DR-DiD. |
| `causal.post_col` | `str | None` | no | `None` | - | causal.method=dr_did | Post-period indicator column. |
| `causal.unit_id_col` | `str | None` | no | `None` | - | causal.method=dr_did, design=panel | Unit id column (panel DR-DiD). |
| `causal.propensity_clip` | `float` | no | `0.01` | - | - | Propensity clipping threshold. |
| `causal.cross_fit` | `bool` | no | `true` | - | - | Enable nuisance cross-fitting. |
| `causal.propensity_calibration` | ``platt` | `isotonic`` | no | `platt` | `platt`, `isotonic` | - | Propensity calibration method. |
| `causal.nuisance_params` | `dict[str, Any]` | no | `{}` | - | - | Nuisance model parameter overrides. |

### Tuning Objective Matrix

#### Non-causal

| task.type | allowed objectives | default |
| --- | --- | --- |
| regression | `rmse`, `mae`, `r2`, `mape` | `rmse` |
| binary | `auc`, `logloss`, `brier`, `accuracy`, `f1`, `precision`, `recall`, `precision_at_k` | `auc` |
| multiclass | `accuracy`, `macro_f1`, `logloss`, `multi_logloss`, `multi_error` | `macro_f1` |
| frontier | `pinball`, `pinball_coverage_penalty` | `pinball` |

#### Causal

| causal.method | allowed objectives | default |
| --- | --- | --- |
| dr | `dr_std_error`, `dr_overlap_penalty`, `dr_balance_priority` | `dr_balance_priority` |
| dr_did | `drdid_std_error`, `drdid_overlap_penalty`, `drdid_balance_priority` | `drdid_balance_priority` |

### Cross-field Constraints (Key Rules)

| group | rule |
| --- | --- |
| version | `config_version` must be `>= 1`. |
| split(timeseries) | `split.time_col` required; `gap>=0`; `embargo>=0`; `test_size>=1` when set. |
| split(blocked timeseries) | `split.train_size>=1` required when `timeseries_mode=blocked`. |
| split(non-timeseries) | `timeseries_mode=expanding`; other timeseries fields are forbidden. |
| split(group) | `split.group_col` required when `split.type=group`. |
| frontier | `frontier.alpha` must satisfy `0<alpha<1`; `split.type=stratified` is forbidden. |
| non-frontier | Custom `frontier.*` is forbidden outside `task.type=frontier`. |
| binary postprocess | `postprocess.*` only for `task.type=binary`; calibration is `platt` only. |
| threshold rules | Fixed threshold and enabled threshold optimization cannot be combined. |
| causal(method=dr) | `task.type` must be `regression|binary`; DiD fields must be unset. |
| causal(method=dr_did) | requires `design/time_col/post_col`; panel also needs `unit_id_col`. |
| causal(binary dr_did) | `causal.estimand` must be `att`. |
| tuning(frontier) | `coverage_target` in `(0,1)`; `coverage_tolerance>=0`; `penalty_weight>=0`. |
| tuning(non-frontier) | `coverage_target` forbidden; tolerance/penalty keep defaults. |
| tuning(causal) | `tuning.objective` must come from causal objective set for selected method. |
| tuning(non-causal) | `tuning.objective` must come from task set; causal knobs keep defaults. |
| auto_num_leaves | requires ratio in `(0,1]`; forbids `lgb_params.num_leaves`. |
| ratio leaf/bin | ratio fields require `0<value<1`; matching absolute `lgb_params` are forbidden. |
| feature_weights | all values must be `>0`; unknown feature names are rejected at training time. |
| top_k | binary-only, `>=1`; required when `tuning.objective=precision_at_k`. |

### Minimal Templates

#### 1) Regression (standard)

```yaml
config_version: 1
task: {type: regression}
data: {path: train.csv, target: target}
split: {type: kfold, n_splits: 5, seed: 42}
export: {artifact_dir: artifacts}
```

#### 2) Binary + calibration

```yaml
config_version: 1
task: {type: binary}
data: {path: train.csv, target: target}
postprocess:
  calibration: platt
  threshold_optimization: {enabled: true, objective: f1}
export: {artifact_dir: artifacts}
```

#### 3) Frontier + coverage-aware tuning

```yaml
config_version: 1
task: {type: frontier}
data: {path: train.csv, target: target}
frontier: {alpha: 0.9}
tuning:
  enabled: true
  objective: pinball_coverage_penalty
  coverage_target: 0.9
  coverage_tolerance: 0.02
  penalty_weight: 2.0
export: {artifact_dir: artifacts}
```

#### 4) Causal DR

```yaml
config_version: 1
task: {type: regression}
data: {path: train.csv, target: outcome}
causal:
  method: dr
  treatment_col: treatment
  estimand: att
  propensity_calibration: platt
export: {artifact_dir: artifacts}
```

#### 5) Causal DR-DiD (panel / repeated_cross_section)

```yaml
config_version: 1
task: {type: regression}
data: {path: train.csv, target: outcome}
causal:
  method: dr_did
  treatment_col: treatment
  design: panel  # or repeated_cross_section
  time_col: time
  post_col: post
  unit_id_col: unit_id  # required for panel
  estimand: att
  propensity_calibration: platt
export: {artifact_dir: artifacts}
```
<!-- RUNCONFIG_REF:END -->

### Phase26.3 Config Notes

`tuning.metrics_candidates` は objective とは独立した候補セットです。task ごとの許可値は以下です。

| task.type | allowed `tuning.metrics_candidates` |
| --- | --- |
| regression | `rmse`, `huber`, `mae` |
| binary | `logloss`, `auc` |
| multiclass | `multi_logloss`, `multi_error` |

## Config Migration

Normalize and validate a RunConfig YAML safely (non-destructive output):

```bash
uv run veldra config migrate --input configs/run.yaml
```

Custom output path:

```bash
uv run veldra config migrate --input configs/run.yaml --output configs/run.normalized.yaml
```

Behavior:
- Strict validation (`RunConfig.model_validate`) is always applied.
- Existing output files are never overwritten.
- Current MVP supports only `config_version=1` and `--target-version 1`.

## Development

Run quality checks:

```bash
uv run ruff check .
uv run pytest -q
```

Core-only tests (exclude GUI):

```bash
uv run pytest -q -m "not gui"
```

GUI-only tests:

```bash
uv run pytest -q -m "gui"
```

Coverage:

```bash
uv run coverage erase
uv run coverage run -m pytest -q
uv run coverage report -m
```

GUI coverage gate (app/services):

```bash
uv run coverage erase
uv run coverage run -m pytest -q tests/test_gui_* tests/test_new_ux.py
uv run coverage report -m src/veldra/gui/app.py src/veldra/gui/services.py
```

Target:
- `src/veldra/gui/app.py` coverage >= 90%
- `src/veldra/gui/services.py` coverage >= 90%

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
