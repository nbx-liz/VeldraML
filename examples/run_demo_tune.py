"""Run end-to-end tuning demo for regression, binary, or multiclass tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:  # pragma: no cover - import path depends on launch style
    from examples.common import (
        DEFAULT_BINARY_DATA_PATH,
        DEFAULT_FRONTIER_DATA_PATH,
        DEFAULT_MULTICLASS_DATA_PATH,
        DEFAULT_OUT_DIR,
        DEFAULT_TARGET,
        format_error,
        make_timestamp_dir,
        save_json,
        save_yaml,
    )
except ModuleNotFoundError:  # pragma: no cover
    from common import (
        DEFAULT_BINARY_DATA_PATH,
        DEFAULT_FRONTIER_DATA_PATH,
        DEFAULT_MULTICLASS_DATA_PATH,
        DEFAULT_OUT_DIR,
        DEFAULT_TARGET,
        format_error,
        make_timestamp_dir,
        save_json,
        save_yaml,
    )
from veldra.api import tune
from veldra.api.exceptions import VeldraValidationError

DEFAULT_DATA_BY_TASK = {
    "regression": Path(__file__).resolve().parent / "data" / "california_housing.csv",
    "binary": DEFAULT_BINARY_DATA_PATH,
    "multiclass": DEFAULT_MULTICLASS_DATA_PATH,
    "frontier": DEFAULT_FRONTIER_DATA_PATH,
}
PREPARE_HINT_BY_TASK = {
    "regression": "prepare_demo_data.py",
    "binary": "prepare_demo_data_binary.py",
    "multiclass": "prepare_demo_data_multiclass.py",
    "frontier": "prepare_demo_data_frontier.py",
}


def _build_dr_demo_frame(rows: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    logits = 0.5 * x1 - 0.4 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    treatment = rng.binomial(1, p)
    y = 2.0 + 1.4 * x1 - 0.7 * x2 + 1.2 * treatment + rng.normal(scale=0.4, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "target": y})


def _build_drdid_demo_frame(n_units: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 60, size=n_units)
    skill = rng.normal(size=n_units)
    p = 1.0 / (1.0 + np.exp(-(-0.6 + 0.03 * (age - 30) + 0.8 * skill)))
    treatment = rng.binomial(1, p, size=n_units)
    base = 9000 + 250 * (age - 30) + 1200 * skill + rng.normal(0, 700, size=n_units)
    pre = base + rng.normal(0, 500, size=n_units)
    post = base + 700 + treatment * 1300 + rng.normal(0, 500, size=n_units)
    pre_df = pd.DataFrame(
        {
            "unit_id": np.arange(n_units),
            "time": 0,
            "post": 0,
            "treatment": treatment,
            "age": age,
            "skill": skill,
            "target": pre,
        }
    )
    post_df = pre_df.copy()
    post_df["time"] = 1
    post_df["post"] = 1
    post_df["target"] = post
    return pd.concat([pre_df, post_df], ignore_index=True)


def _is_causal_objective(objective: str | None) -> bool:
    obj = (objective or "").lower()
    return obj.startswith("dr_") or obj.startswith("drdid_")


def _resolve_causal_method(args: argparse.Namespace) -> str | None:
    if args.causal_method is not None:
        return args.causal_method
    objective = (args.objective or "").lower()
    if objective.startswith("drdid_"):
        return "dr_did"
    if objective.startswith("dr_"):
        return "dr"
    return None


def _ensure_default_causal_demo_data(
    args: argparse.Namespace,
    out_root: Path,
) -> Path | None:
    if args.data_path is not None or not _is_causal_objective(args.objective):
        return None
    method = _resolve_causal_method(args)
    if method is None:
        return None

    out_root.mkdir(parents=True, exist_ok=True)
    if method == "dr":
        path = out_root / "causal_dr_tune_demo.csv"
        if not path.exists():
            _build_dr_demo_frame().to_csv(path, index=False)
        return path
    path = out_root / "causal_drdid_tune_demo.csv"
    if not path.exists():
        _build_drdid_demo_frame().to_csv(path, index=False)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["regression", "binary", "multiclass", "frontier"],
        default="regression",
        help="Task type for tuning demo.",
    )
    parser.add_argument("--data-path", default=None, help="Input CSV/Parquet path.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output root directory (default: examples/out).",
    )
    parser.add_argument("--n-trials", type=int, default=5, help="Number of tuning trials.")
    parser.add_argument("--objective", default=None, help="Task-allowed objective metric.")
    parser.add_argument(
        "--causal-method",
        choices=["dr", "dr_did"],
        default=None,
        help="Enable causal tuning mode and choose method.",
    )
    parser.add_argument(
        "--treatment-col",
        default="treatment",
        help="Treatment column name for causal tuning.",
    )
    parser.add_argument(
        "--causal-design",
        choices=["panel", "repeated_cross_section"],
        default="panel",
        help="DR-DiD design when --causal-method=dr_did.",
    )
    parser.add_argument("--time-col", default="time", help="DR-DiD time column.")
    parser.add_argument("--post-col", default="post", help="DR-DiD post indicator column.")
    parser.add_argument("--unit-id-col", default="unit_id", help="DR-DiD panel unit id column.")
    parser.add_argument(
        "--causal-penalty-weight",
        type=float,
        default=None,
        help="Causal objective penalty weight.",
    )
    parser.add_argument(
        "--causal-balance-threshold",
        type=float,
        default=None,
        help="Causal balance threshold for balance-priority objectives.",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=None,
        help="Frontier-only target coverage for pinball_coverage_penalty objective.",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=None,
        help="Frontier-only tolerance band around target coverage.",
    )
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=None,
        help="Frontier-only penalty weight for coverage deviation.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume existing study if present.")
    parser.add_argument("--study-name", default=None, help="Explicit Optuna study name.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Tuning log level.",
    )
    parser.add_argument(
        "--search-space-file",
        default=None,
        help="Optional JSON/YAML file describing tuning.search_space.",
    )
    return parser.parse_args(argv)


def _load_search_space(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        raise VeldraValidationError(f"search-space file not found: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise VeldraValidationError("search-space file must contain a mapping object.")
    return payload


def _build_tune_config(
    args: argparse.Namespace,
    data_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    split_type = "kfold" if args.task in {"regression", "frontier"} else "stratified"
    tuning_cfg: dict[str, Any] = {
        "enabled": True,
        "n_trials": args.n_trials,
        "preset": "fast",
        "resume": bool(args.resume),
        "log_level": args.log_level,
    }
    if args.objective:
        tuning_cfg["objective"] = args.objective
    if args.causal_penalty_weight is not None:
        tuning_cfg["causal_penalty_weight"] = args.causal_penalty_weight
    if args.causal_balance_threshold is not None:
        tuning_cfg["causal_balance_threshold"] = args.causal_balance_threshold
    if args.coverage_target is not None:
        tuning_cfg["coverage_target"] = args.coverage_target
    if args.coverage_tolerance is not None:
        tuning_cfg["coverage_tolerance"] = args.coverage_tolerance
    if args.penalty_weight is not None:
        tuning_cfg["penalty_weight"] = args.penalty_weight
    if args.study_name:
        tuning_cfg["study_name"] = args.study_name

    search_space = _load_search_space(args.search_space_file)
    if search_space:
        tuning_cfg["search_space"] = search_space

    inferred_causal_method = _resolve_causal_method(args)

    config: dict[str, Any] = {
        "config_version": 1,
        "task": {"type": args.task},
        "data": {"path": str(data_path), "target": DEFAULT_TARGET},
        "split": {"type": split_type, "n_splits": 3, "seed": 42},
        "train": {"seed": 42},
        "tuning": tuning_cfg,
        "export": {"artifact_dir": str(artifact_dir)},
    }
    if inferred_causal_method is not None:
        causal_cfg: dict[str, Any] = {
            "method": inferred_causal_method,
            "treatment_col": args.treatment_col,
        }
        if inferred_causal_method == "dr_did":
            causal_cfg.update(
                {
                    "design": args.causal_design,
                    "time_col": args.time_col,
                    "post_col": args.post_col,
                }
            )
            if args.causal_design == "panel":
                causal_cfg["unit_id_col"] = args.unit_id_col
        config["causal"] = causal_cfg
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default_demo_root = Path(__file__).resolve().parent / "data"
    causal_default = _ensure_default_causal_demo_data(args, default_demo_root)
    data_path = (
        Path(args.data_path)
        if args.data_path
        else (causal_default if causal_default is not None else DEFAULT_DATA_BY_TASK[args.task])
    )
    if not data_path.exists():
        prepare_script = PREPARE_HINT_BY_TASK[args.task]
        raise SystemExit(
            f"Input data was not found: {data_path}\n"
            f"Hint: run `uv run python examples/{prepare_script}` first."
        )

    try:
        run_dir = make_timestamp_dir(args.out_dir)
        # Keep per-run outputs clean while preserving a stable location for resume.
        artifact_root = data_path.resolve().parent / ".veldra_tuning_artifacts"
        config = _build_tune_config(
            args=args,
            data_path=data_path,
            artifact_dir=artifact_root,
        )
        if args.study_name is None and not args.resume:
            config["tuning"]["study_name"] = f"demo_{args.task}_{run_dir.name}"
        tune_result = tune(config)
        save_json(run_dir / "tune_result.json", tune_result)
        save_yaml(run_dir / "used_config.yaml", config)
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(
            format_error(
                exc,
                "Check objective, search-space schema, and resume/study-name settings.",
            )
        ) from exc

    print(f"run_id: {tune_result.run_id}")
    print(f"task_type: {tune_result.task_type}")
    print(f"best_score: {tune_result.best_score:.6f}")
    print(f"best_params: {tune_result.best_params}")
    print(f"summary_path: {tune_result.metadata['summary_path']}")
    print(f"trials_path: {tune_result.metadata['trials_path']}")
    print(f"storage_url: {tune_result.metadata['storage_url']}")
    print(f"output_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
