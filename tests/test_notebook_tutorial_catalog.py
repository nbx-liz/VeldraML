from __future__ import annotations

import json
from pathlib import Path

TUTORIAL_NOTEBOOKS = [
    "tutorial_00_quickstart.ipynb",
    "tutorial_01_regression_basics.ipynb",
    "tutorial_02_binary_classification_tuning.ipynb",
    "tutorial_03_frontier_quantile_regression.ipynb",
    "tutorial_04_scenario_simulation.ipynb",
    "tutorial_05_causal_dr_lalonde.ipynb",
    "tutorial_06_causal_drdid_lalonde.ipynb",
    "tutorial_07_model_evaluation_guide.ipynb",
]

REMOVED_LEGACY_WORKFLOW_NOTEBOOKS = [
    "regression_analysis_workflow.ipynb",
    "binary_tuning_analysis_workflow.ipynb",
    "frontier_analysis_workflow.ipynb",
    "simulate_analysis_workflow.ipynb",
    "lalonde_dr_analysis_workflow.ipynb",
    "lalonde_drdid_analysis_workflow.ipynb",
]

REQUIRED_EDUCATION_SECTIONS = [
    "## Concept Primer",
    "## Config Guide",
    "## Result Interpretation",
    "## If-Then Sensitivity",
    "## Common Pitfalls",
    "## Further Reading",
]


def _source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        src = cell.get("source", [])
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def test_tutorial_notebooks_exist_and_include_education_sections() -> None:
    for notebook in TUTORIAL_NOTEBOOKS:
        path = Path("notebooks/tutorials") / notebook
        assert path.exists(), notebook
        source = _source(path)
        for marker in REQUIRED_EDUCATION_SECTIONS:
            assert marker in source, f"{notebook}: missing {marker}"


def test_legacy_workflow_notebooks_are_removed() -> None:
    for notebook in REMOVED_LEGACY_WORKFLOW_NOTEBOOKS:
        assert not (Path("notebooks") / notebook).exists(), notebook
