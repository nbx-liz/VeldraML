# Phase26.2 Parity Report

## Scope
- Parity criterion: `reachability_and_artifact_outputs`
- Source of notebook evidence: `notebooks/phase26_2_execution_manifest.json`
- Source of GUI evidence: `tests/e2e_playwright/test_uc*.py`

## UC Matrix
| UC | Notebook evidence | GUI evidence | Parity status | Notes |
|---|---|---|---|---|
| UC-1 | `notebooks/phase26_2_uc01_regression_fit_evaluate.ipynb` | `tests/e2e_playwright/test_uc01_regression_flow.py` | pass | End-to-end page flow and artifact evidence recorded. |
| UC-2 | `notebooks/phase26_2_uc02_binary_tune_evaluate.ipynb` | `tests/e2e_playwright/test_uc02_binary_tune_flow.py` | pass | Tune + evaluate executed; train tuning controls validated in GUI. |
| UC-3 | `notebooks/phase26_2_uc03_frontier_fit_evaluate.ipynb` | `tests/e2e_playwright/test_uc03_frontier_flow.py` | pass | Frontier alpha guidance validated in GUI. |
| UC-4 | `notebooks/phase26_2_uc04_causal_dr_estimate.ipynb` | `tests/e2e_playwright/test_uc04_causal_dr_flow.py` | pass | DR estimate evidence + Target causal guidance path. |
| UC-5 | `notebooks/phase26_2_uc05_causal_drdid_estimate.ipynb` | `tests/e2e_playwright/test_uc05_causal_drdid_flow.py` | pass | DR-DiD panel evidence + GUI method guidance path. |
| UC-6 | `notebooks/phase26_2_uc06_causal_dr_tune.ipynb` | `tests/e2e_playwright/test_uc06_causal_tune_flow.py` | pass | Causal objective tuning evidence + train controls. |
| UC-7 | `notebooks/phase26_2_uc07_artifact_evaluate.ipynb` | `tests/e2e_playwright/test_uc07_evaluate_existing_artifact_flow.py` | pass | Existing artifact evaluation path tested in Results page. |
| UC-8 | `notebooks/phase26_2_uc08_artifact_reevaluate.ipynb` | `tests/e2e_playwright/test_uc08_reevaluate_flow.py` | pass | Schema precheck and re-evaluate path covered. |
| UC-9 | `notebooks/phase26_2_uc09_export_python_onnx.ipynb` | `tests/e2e_playwright/test_uc09_export_python_onnx_flow.py` | pass | Python/ONNX export evidence and run-page controls covered. |
| UC-10 | `notebooks/phase26_2_uc10_export_html_excel.ipynb` | `tests/e2e_playwright/test_uc10_export_html_excel_flow.py` | pass | HTML/Excel export path covered; optional dependency degradation documented. |

## Optional Dependency Degradation
- UC-10 Excel report export may degrade when `openpyxl` is unavailable.
- This behavior is recorded in manifest and treated as expected graceful degradation.

## CI Policy
- `gui_smoke`: UC-1, UC-8, UC-10
- `gui_e2e`: full UC-1..UC-10 suite (manual/full run)
