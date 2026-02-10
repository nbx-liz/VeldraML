"""Run simulation demo against a trained regression artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

try:  # pragma: no cover - import path depends on launch style
    from examples.common import (
        DEFAULT_DATA_PATH,
        DEFAULT_OUT_DIR,
        DEFAULT_TARGET,
        format_error,
        make_timestamp_dir,
        save_json,
        save_yaml,
    )
except ModuleNotFoundError:  # pragma: no cover
    from common import (
        DEFAULT_DATA_PATH,
        DEFAULT_OUT_DIR,
        DEFAULT_TARGET,
        format_error,
        make_timestamp_dir,
        save_json,
        save_yaml,
    )

from veldra.api import Artifact, fit, simulate
from veldra.api.exceptions import VeldraValidationError
from veldra.data import load_tabular_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Input CSV path prepared by prepare_demo_data.py.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output root directory (default: examples/out).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="CV fold count.")
    return parser.parse_args(argv)


def _build_scenarios(frame: Any, target_col: str) -> list[dict[str, Any]]:
    feature_cols = [
        col for col in frame.columns if col != target_col and frame[col].dtype.kind in "biufc"
    ]
    if not feature_cols:
        raise VeldraValidationError("Simulation demo requires at least one numeric feature column.")
    first = feature_cols[0]
    second = feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    return [
        {"name": "shift_first_feature", "actions": [{"op": "add", "column": first, "value": 0.5}]},
        {
            "name": "scale_second_feature",
            "actions": [{"op": "mul", "column": second, "value": 1.1}],
        },
    ]


def build_run_config(
    train_path: Path,
    artifact_dir: Path,
    seed: int,
    n_splits: int,
) -> dict[str, Any]:
    return {
        "config_version": 1,
        "task": {"type": "regression"},
        "data": {"path": str(train_path), "target": DEFAULT_TARGET},
        "split": {"type": "kfold", "n_splits": n_splits, "seed": seed},
        "train": {"seed": seed, "early_stopping_rounds": 50},
        "export": {"artifact_dir": str(artifact_dir)},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(
            f"Input data was not found: {data_path}\n"
            "Hint: run `uv run python examples/prepare_demo_data.py` first."
        )

    try:
        frame = load_tabular_data(str(data_path))
        if frame.empty:
            raise VeldraValidationError("Simulation demo input data is empty.")
        if DEFAULT_TARGET not in frame.columns:
            raise VeldraValidationError(
                f"Simulation demo input must include target column '{DEFAULT_TARGET}'."
            )
        run_dir = make_timestamp_dir(args.out_dir)
        train_df, test_df = train_test_split(frame, test_size=0.2, random_state=args.seed)
        if train_df.empty or test_df.empty:
            raise VeldraValidationError("Train/test split produced an empty partition.")

        train_path = run_dir / "train_data_simulate.csv"
        train_df.to_csv(train_path, index=False)
        config = build_run_config(
            train_path=train_path,
            artifact_dir=run_dir / "artifacts",
            seed=args.seed,
            n_splits=args.n_splits,
        )

        run_result = fit(config)
        artifact = Artifact.load(run_result.artifact_path)
        scenarios = _build_scenarios(test_df, DEFAULT_TARGET)
        sim_result = simulate(artifact, test_df, scenarios)

        sim_result.data.to_csv(run_dir / "simulate_result.csv", index=False)
        save_json(
            run_dir / "simulate_summary.json",
            {
                "run_id": run_result.run_id,
                "artifact_path": run_result.artifact_path,
                "simulation_metadata": sim_result.metadata,
            },
        )
        save_yaml(run_dir / "used_scenarios.yaml", {"scenarios": scenarios})
    except Exception as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(
            format_error(
                exc,
                "Check data format and simulation action columns before retry.",
            )
        ) from exc

    print(f"run_id: {run_result.run_id}")
    print(f"artifact_path: {run_result.artifact_path}")
    print(f"n_rows: {sim_result.metadata['n_rows']}")
    print(f"n_scenarios: {sim_result.metadata['n_scenarios']}")
    print(f"output_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
