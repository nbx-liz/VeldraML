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
- Hyperparameter tuning workflow: `tune` (regression/binary/multiclass)
- Scenario simulation workflow: `simulate` (regression/binary/multiclass/frontier)
- Export workflow: `export` (`python` + optional `onnx`)

## Project Status

Implemented:
- `fit`, `predict`, `evaluate` for `task.type=regression`
- `fit`, `predict`, `evaluate` for `task.type=binary`
- `fit`, `predict`, `evaluate` for `task.type=multiclass`
- `fit`, `predict`, `evaluate` for `task.type=frontier`
- `tune` for `task.type=regression|binary|multiclass` (Optuna-based MVP)
- `simulate` for `task.type=regression|binary|multiclass|frontier` (Scenario DSL MVP)
- `export` for all implemented tasks (`python` always, `onnx` optional dependency)

Not implemented yet:
- `tune` for `task.type=frontier`
- threshold optimization is optional for binary classification (default is fixed `0.5`)

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

Simulate demo (regression baseline with scenario actions):

```bash
uv run python examples/run_demo_simulate.py --data-path examples/data/california_housing.csv
```

Export demo:

```bash
uv run python examples/run_demo_export.py --artifact-path <artifact_dir> --format python
```

ONNX export (requires optional dependencies):

```bash
uv sync --extra export-onnx
uv run python examples/run_demo_export.py --artifact-path <artifact_dir> --format onnx
```

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
