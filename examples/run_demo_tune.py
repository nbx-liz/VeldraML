"""Run end-to-end tuning demo for regression, binary, or multiclass tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:  # pragma: no cover - import path depends on launch style
    from examples.common import (
        DEFAULT_BINARY_DATA_PATH,
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
}
PREPARE_HINT_BY_TASK = {
    "regression": "prepare_demo_data.py",
    "binary": "prepare_demo_data_binary.py",
    "multiclass": "prepare_demo_data_multiclass.py",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["regression", "binary", "multiclass"],
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
    split_type = "kfold" if args.task == "regression" else "stratified"
    tuning_cfg: dict[str, Any] = {
        "enabled": True,
        "n_trials": args.n_trials,
        "preset": "fast",
        "resume": bool(args.resume),
        "log_level": args.log_level,
    }
    if args.objective:
        tuning_cfg["objective"] = args.objective
    if args.study_name:
        tuning_cfg["study_name"] = args.study_name

    search_space = _load_search_space(args.search_space_file)
    if search_space:
        tuning_cfg["search_space"] = search_space

    return {
        "config_version": 1,
        "task": {"type": args.task},
        "data": {"path": str(data_path), "target": DEFAULT_TARGET},
        "split": {"type": split_type, "n_splits": 3, "seed": 42},
        "train": {"seed": 42},
        "tuning": tuning_cfg,
        "export": {"artifact_dir": str(artifact_dir)},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data_path) if args.data_path else DEFAULT_DATA_BY_TASK[args.task]
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
