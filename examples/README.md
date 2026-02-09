# Examples

These scripts run the current regression workflow (`fit` / `predict` / `evaluate`)
on California Housing data.

## 1) Prepare local CSV

```bash
uv run python examples/prepare_demo_data.py
```

This creates `examples/data/california_housing.csv`.

## 2) Run end-to-end demo

```bash
uv run python examples/run_demo_regression.py
```

Outputs are written to `examples/out/<timestamp>/`:

- `run_result.json`
- `eval_result.json`
- `predictions_sample.csv`
- `used_config.yaml`
- artifact files under `artifacts/<run_id>/`

## 3) Re-evaluate an existing artifact

```bash
uv run python examples/evaluate_demo_artifact.py --artifact-path <artifact_dir>
```

This writes `eval_only_result.json` to `examples/out/<timestamp>/`.

## Binary examples

These scripts run the binary workflow on sklearn Breast Cancer data.

### 1) Prepare local CSV

```bash
uv run python examples/prepare_demo_data_binary.py
```

This creates `examples/data/breast_cancer_binary.csv`.

### 2) Run end-to-end binary demo

```bash
uv run python examples/run_demo_binary.py
```

Outputs are written to `examples/out/<timestamp>/`:

- `run_result.json`
- `eval_result.json`
- `predictions_sample.csv`
- `used_config.yaml`
- artifact files under `artifacts/<run_id>/`

### 3) Re-evaluate an existing binary artifact

```bash
uv run python examples/evaluate_demo_binary_artifact.py --artifact-path <artifact_dir>
```

This writes `eval_only_result.json` to `examples/out/<timestamp>/`.
