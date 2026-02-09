# VeldraML

VeldraML is a config-driven LightGBM analysis library.
Current implemented tasks are regression and binary classification.

## Features

- Stable public API under `veldra.api.*`
- Config-driven execution via `RunConfig`
- Artifact-based reproducibility (`save/load/predict/evaluate`)
- Regression workflow: `fit`, `predict`, `evaluate`
- Binary workflow: `fit`, `predict`, `evaluate` with OOF-based Platt calibration

## Project Status

Implemented:
- `fit`, `predict`, `evaluate` for `task.type=regression`
- `fit`, `predict`, `evaluate` for `task.type=binary`

Not implemented yet:
- `tune`, `simulate`, `export`
- multiclass/frontier training
- threshold optimization for binary classification (fixed `0.5` in current phase)

## Requirements

- Python `>=3.11`
- Dependency management: `uv`

## Installation

### Development install

```bash
uv sync --dev
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

Example outputs are saved under `examples/out/<timestamp>/`:

- `run_result.json`
- `eval_result.json`
- `predictions_sample.csv`
- `used_config.yaml`
- `artifacts/<run_id>/...`

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
examples/       # runnable demos
tests/          # unit tests
```

## Notes

- Root `README.md` is the single source of truth for project documentation.
- Design decisions and phase logs are tracked in `DESIGN_BLUEPRINT.md` and `HISTORY.md`.
