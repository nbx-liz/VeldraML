#!/usr/bin/env python3
"""Generate RunConfig reference section for README.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

from veldra.config import models

START = "<!-- RUNCONFIG_REF:START -->"
END = "<!-- RUNCONFIG_REF:END -->"

DESCRIPTION: dict[str, str] = {
    "config_version": "Configuration schema version.",
    "task": "Task-level settings container.",
    "task.type": "Task kind to execute.",
    "data": "Input data contract settings.",
    "data.path": "Training data path (required for fit/tune/estimate_dr).",
    "data.target": "Target column name.",
    "data.id_cols": "Identifier columns protected in simulation.",
    "data.categorical": "Categorical feature columns for LightGBM.",
    "data.drop_cols": "Columns dropped before training.",
    "split": "Cross-validation split settings.",
    "split.type": "Split strategy selector.",
    "split.n_splits": "Number of CV folds.",
    "split.time_col": "Time column for timeseries split.",
    "split.group_col": "Group column for group split.",
    "split.seed": "Random seed for split shuffling.",
    "split.timeseries_mode": "Timeseries CV mode.",
    "split.test_size": "Validation window size for timeseries split.",
    "split.gap": "Gap between train and validation windows.",
    "split.embargo": "Embargo window after validation horizon.",
    "split.train_size": "Fixed train window size in blocked mode.",
    "train": "Model training settings.",
    "train.lgb_params": "LightGBM parameter overrides.",
    "train.early_stopping_rounds": "Early stopping rounds.",
    "train.seed": "Training seed.",
    "tuning": "Hyperparameter tuning settings.",
    "tuning.enabled": "Enable/disable tuning path.",
    "tuning.n_trials": "Optuna trial count.",
    "tuning.search_space": "Explicit search space spec.",
    "tuning.preset": "Default search space preset.",
    "tuning.objective": "Objective metric name.",
    "tuning.resume": "Resume an existing study.",
    "tuning.study_name": "Explicit study name.",
    "tuning.log_level": "Tuning log verbosity.",
    "tuning.coverage_target": "Target coverage for frontier tuning.",
    "tuning.coverage_tolerance": "Allowed coverage deviation.",
    "tuning.penalty_weight": "Coverage penalty weight.",
    "tuning.causal_penalty_weight": "Penalty weight for causal objectives.",
    "tuning.causal_balance_threshold": "Max weighted SMD threshold.",
    "postprocess": "Postprocessing settings.",
    "postprocess.calibration": "Probability calibration method.",
    "postprocess.threshold": "Fixed decision threshold for binary.",
    "postprocess.threshold_optimization": "Threshold optimization settings.",
    "postprocess.threshold_optimization.enabled": "Enable threshold search.",
    "postprocess.threshold_optimization.objective": "Threshold objective.",
    "simulation": "Simulation DSL settings.",
    "simulation.scenarios": "Scenario list.",
    "simulation.actions": "Action list (legacy/helper).",
    "simulation.constraints": "Constraint list (reserved).",
    "export": "Artifact/export settings.",
    "export.artifact_dir": "Artifact root directory.",
    "export.inference_package": "Reserved package export toggle.",
    "export.onnx_optimization": "ONNX optimization settings.",
    "export.onnx_optimization.enabled": "Enable ONNX optimization.",
    "export.onnx_optimization.mode": "ONNX optimization mode.",
    "frontier": "Frontier task settings.",
    "frontier.alpha": "Quantile alpha (0,1).",
    "causal": "Causal estimation settings.",
    "causal.method": "Causal estimator family.",
    "causal.treatment_col": "Treatment column name.",
    "causal.estimand": "Target estimand.",
    "causal.design": "DR-DiD design type.",
    "causal.time_col": "Time column for DR-DiD.",
    "causal.post_col": "Post-period indicator column.",
    "causal.unit_id_col": "Unit id column (panel DR-DiD).",
    "causal.propensity_clip": "Propensity clipping threshold.",
    "causal.cross_fit": "Enable nuisance cross-fitting.",
    "causal.propensity_calibration": "Propensity calibration method.",
    "causal.nuisance_params": "Nuisance model parameter overrides.",
}

SCOPE: dict[str, str] = {
    "frontier": "task.type=frontier",
    "frontier.alpha": "task.type=frontier",
    "postprocess.calibration": "task.type=binary",
    "postprocess.threshold": "task.type=binary",
    "postprocess.threshold_optimization": "task.type=binary",
    "postprocess.threshold_optimization.enabled": "task.type=binary",
    "postprocess.threshold_optimization.objective": "task.type=binary",
    "tuning.coverage_target": "task.type=frontier",
    "tuning.coverage_tolerance": "task.type=frontier",
    "tuning.penalty_weight": "task.type=frontier",
    "tuning.causal_penalty_weight": "causal configured",
    "tuning.causal_balance_threshold": "causal configured",
    "causal": "task.type in {regression,binary}",
    "causal.design": "causal.method=dr_did",
    "causal.time_col": "causal.method=dr_did",
    "causal.post_col": "causal.method=dr_did",
    "causal.unit_id_col": "causal.method=dr_did, design=panel",
}


def _is_model(annotation: Any) -> type[BaseModel] | None:
    origin = get_origin(annotation)
    if origin in (Union, getattr(__import__("types"), "UnionType", Union)):
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _format_annotation(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Literal:
        values = [f"`{v}`" for v in get_args(annotation)]
        return " | ".join(values)
    if origin in (list,):
        args = get_args(annotation)
        inner = _format_annotation(args[0]) if args else "Any"
        return f"list[{inner}]"
    if origin in (dict,):
        args = get_args(annotation)
        if len(args) == 2:
            return f"dict[{_format_annotation(args[0])}, {_format_annotation(args[1])}]"
        return "dict[Any, Any]"
    if origin in (Union, getattr(__import__("types"), "UnionType", Union)):
        args = [a for a in get_args(annotation)]
        if type(None) in args:
            args.remove(type(None))
            if len(args) == 1:
                return f"{_format_annotation(args[0])} | None"
        return " | ".join(_format_annotation(a) for a in args)
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _allowed_values(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Literal:
        return ", ".join(f"`{v}`" for v in get_args(annotation))
    if origin in (Union, getattr(__import__("types"), "UnionType", Union)):
        values: list[str] = []
        for arg in get_args(annotation):
            if get_origin(arg) is Literal:
                values.extend(f"`{v}`" for v in get_args(arg))
        return ", ".join(values) if values else "-"
    return "-"


def _format_default(field: Any) -> str:
    if field.is_required():
        return "-"
    if field.default_factory is not None:
        if field.default_factory is list:
            return "`[]`"
        if field.default_factory is dict:
            return "`{}`"
        return "`<factory>`"
    if field.default is PydanticUndefined:
        return "-"
    value = field.default
    if isinstance(value, str):
        return f"`{value}`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if value is None:
        return "`None`"
    return f"`{value}`"


def _iter_rows(prefix: str, model: type[BaseModel]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        annotation = field.annotation
        rows.append(
            {
                "path": path,
                "type": _format_annotation(annotation),
                "required": "yes" if field.is_required() else "no",
                "default": _format_default(field),
                "allowed_values": _allowed_values(annotation),
                "scope": SCOPE.get(path, "-"),
                "description": DESCRIPTION.get(path, f"{path} setting."),
            }
        )
        nested = _is_model(annotation)
        if nested is not None:
            rows.extend(_iter_rows(path, nested))
    return rows


def _render_table(rows: list[dict[str, str]]) -> str:
    header = (
        "| path | type | required | default | allowed values | scope | description |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    body = "\n".join(
        [
            f"| `{r['path']}` | `{r['type']}` | {r['required']} | {r['default']} | "
            f"{r['allowed_values']} | {r['scope']} | {r['description']} |"
            for r in rows
        ]
    )
    return header + body + "\n"


def _objective_matrix() -> str:
    regular = (
        "| task.type | allowed objectives | default |\n"
        "| --- | --- | --- |\n"
        "| regression | `rmse`, `mae`, `r2` | `rmse` |\n"
        "| binary | `auc`, `logloss`, `brier`, `accuracy`, `f1`, `precision`, `recall` | `auc` |\n"
        "| multiclass | `accuracy`, `macro_f1`, `logloss` | `macro_f1` |\n"
        "| frontier | `pinball`, `pinball_coverage_penalty` | `pinball` |\n"
    )
    causal = (
        "| causal.method | allowed objectives | default |\n"
        "| --- | --- | --- |\n"
        "| dr | `dr_std_error`, `dr_overlap_penalty`, `dr_balance_priority` | "
        "`dr_balance_priority` |\n"
        "| dr_did | `drdid_std_error`, `drdid_overlap_penalty`, "
        "`drdid_balance_priority` | `drdid_balance_priority` |\n"
    )
    return (
        "### Tuning Objective Matrix\n\n"
        "#### Non-causal\n\n"
        f"{regular}\n"
        "#### Causal\n\n"
        f"{causal}\n"
    )


def _cross_constraints() -> str:
    return """### Cross-field Constraints (Key Rules)

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

"""


def _templates() -> str:
    return """### Minimal Templates

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
"""


def generate_reference_markdown() -> str:
    rows = _iter_rows("", models.RunConfig)
    header = "### Field Reference\n\n"
    note = (
        "_This section is auto-generated from `src/veldra/config/models.py`. "
        "Do not edit manually._\n\n"
    )
    return (
        header
        + note
        + _render_table(rows)
        + "\n"
        + _objective_matrix()
        + _cross_constraints()
        + _templates()
    )


def _replace_block(readme: str, content: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError(f"README markers not found: {START} / {END}")
    before, rest = readme.split(START, 1)
    _, after = rest.split(END, 1)
    return f"{before}{START}\n{content.rstrip()}\n{END}{after}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RunConfig README reference block.")
    parser.add_argument("--readme", default="README.md", help="README path.")
    parser.add_argument("--check", action="store_true", help="Fail if block is out-of-date.")
    parser.add_argument("--write", action="store_true", help="Write generated content into README.")
    args = parser.parse_args()

    readme_path = Path(args.readme)
    readme_text = readme_path.read_text(encoding="utf-8")
    generated = generate_reference_markdown()
    expected = _replace_block(readme_text, generated)

    if args.check:
        if readme_text != expected:
            print("README RunConfig reference block is out of date.", file=sys.stderr)
            return 1
        return 0

    if args.write:
        readme_path.write_text(expected, encoding="utf-8")
        return 0

    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
