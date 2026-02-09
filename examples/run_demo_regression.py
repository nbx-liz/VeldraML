"""Run end-to-end regression demo with fit/predict/evaluate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - import path depends on how the script is launched
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
from veldra.api import Artifact, evaluate, fit, predict
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
            raise VeldraValidationError("Demo input data is empty.")
        if DEFAULT_TARGET not in frame.columns:
            raise VeldraValidationError(
                f"Demo input must include target column '{DEFAULT_TARGET}'."
            )

        train_df, test_df = train_test_split(
            frame,
            test_size=0.2,
            random_state=args.seed,
        )
        if train_df.empty or test_df.empty:
            raise VeldraValidationError("Train/test split produced an empty partition.")

        run_dir = make_timestamp_dir(args.out_dir)
        train_path = run_dir / "train_data.csv"
        train_df.to_csv(train_path, index=False)
        config = build_run_config(
            train_path=train_path,
            artifact_dir=run_dir / "artifacts",
            seed=args.seed,
            n_splits=args.n_splits,
        )

        run_result = fit(config)
        artifact = Artifact.load(run_result.artifact_path)
        eval_result = evaluate(artifact, test_df)

        pred = predict(artifact, test_df.drop(columns=[DEFAULT_TARGET]))
        pred_sample = pd.DataFrame(
            {
                DEFAULT_TARGET: test_df[DEFAULT_TARGET].to_numpy()[:20],
                "prediction": pred.data[:20],
            }
        )

        save_json(run_dir / "run_result.json", run_result)
        save_json(run_dir / "eval_result.json", eval_result)
        pred_sample.to_csv(run_dir / "predictions_sample.csv", index=False)
        save_yaml(run_dir / "used_config.yaml", config)
    except Exception as exc:  # pragma: no cover - exercised via CLI failure only
        raise SystemExit(
            format_error(
                exc,
                "Check data format, target column, and split settings before re-running.",
            )
        ) from exc

    print(f"run_id: {run_result.run_id}")
    print(f"artifact_path: {run_result.artifact_path}")
    print(f"rmse: {eval_result.metrics['rmse']:.6f}")
    print(f"mae: {eval_result.metrics['mae']:.6f}")
    print(f"r2: {eval_result.metrics['r2']:.6f}")
    print(f"output_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
