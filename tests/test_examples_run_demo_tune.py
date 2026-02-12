from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from examples import run_demo_tune


def _regression_frame(rows: int = 28, seed: int = 901) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    y = 2.0 * x1 - 1.1 * x2 + rng.normal(scale=0.3, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _binary_frame(rows: int = 30, seed: int = 902) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=rows)
    x2 = rng.normal(size=rows)
    score = x1 - x2 + rng.normal(scale=0.35, size=rows)
    y = (score > np.median(score)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _multiclass_frame(rows_per_class: int = 10, seed: int = 903) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = ["a", "b", "c"]
    frames: list[pd.DataFrame] = []
    for idx, label in enumerate(labels):
        center = idx * 1.5
        x1 = rng.normal(loc=center, scale=0.4, size=rows_per_class)
        x2 = rng.normal(loc=1.5 - center, scale=0.4, size=rows_per_class)
        frames.append(pd.DataFrame({"x1": x1, "x2": x2, "target": label}))
    return pd.concat(frames, ignore_index=True)


def _frontier_frame(rows: int = 30, seed: int = 904) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, size=rows)
    x2 = rng.normal(size=rows)
    y = 1.4 + 1.2 * x1 - 0.4 * x2 + rng.normal(scale=0.2, size=rows)
    y = y + rng.exponential(scale=0.15, size=rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _data_for_task(task: str) -> pd.DataFrame:
    if task == "regression":
        return _regression_frame()
    if task == "binary":
        return _binary_frame()
    if task == "frontier":
        return _frontier_frame()
    return _multiclass_frame()


def test_run_demo_tune_supports_all_tasks(tmp_path) -> None:
    for task in ("regression", "binary", "multiclass", "frontier"):
        data_path = tmp_path / f"{task}.csv"
        out_dir = tmp_path / f"out_{task}"
        _data_for_task(task).to_csv(data_path, index=False)

        exit_code = run_demo_tune.main(
            [
                "--task",
                task,
                "--data-path",
                str(data_path),
                "--out-dir",
                str(out_dir),
                "--n-trials",
                "1",
                "--log-level",
                "INFO",
            ]
        )
        assert exit_code == 0
        run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert (run_dir / "tune_result.json").exists()
        assert (run_dir / "used_config.yaml").exists()


def test_run_demo_tune_applies_search_space_and_resume(tmp_path) -> None:
    data_path = tmp_path / "regression.csv"
    out_dir = tmp_path / "out_resume"
    _regression_frame().to_csv(data_path, index=False)

    search_space_path = tmp_path / "space.yaml"
    search_space_path.write_text(
        yaml.safe_dump(
            {
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.05},
                "num_leaves": {"type": "int", "low": 8, "high": 16},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    first_code = run_demo_tune.main(
        [
            "--task",
            "regression",
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--n-trials",
            "1",
            "--study-name",
            "demo_resume",
            "--search-space-file",
            str(search_space_path),
        ]
    )
    assert first_code == 0
    first_run_dir = sorted([p for p in out_dir.iterdir() if p.is_dir()])[0]
    first_result = json.loads((first_run_dir / "tune_result.json").read_text(encoding="utf-8"))
    first_trials = pd.read_parquet(first_result["metadata"]["trials_path"])
    assert len(first_trials) == 1

    second_code = run_demo_tune.main(
        [
            "--task",
            "regression",
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--n-trials",
            "2",
            "--study-name",
            "demo_resume",
            "--resume",
            "--search-space-file",
            str(search_space_path),
            "--log-level",
            "DEBUG",
        ]
    )
    assert second_code == 0
    second_run_dir = sorted([p for p in out_dir.iterdir() if p.is_dir()])[-1]
    second_result = json.loads((second_run_dir / "tune_result.json").read_text(encoding="utf-8"))
    second_trials = pd.read_parquet(second_result["metadata"]["trials_path"])
    assert len(second_trials) >= 3

    used_config = yaml.safe_load((second_run_dir / "used_config.yaml").read_text(encoding="utf-8"))
    assert used_config["tuning"]["search_space"]
    assert used_config["tuning"]["resume"] is True
    assert used_config["tuning"]["log_level"] == "DEBUG"


def test_run_demo_tune_frontier_accepts_coverage_penalty_options(tmp_path) -> None:
    data_path = tmp_path / "frontier.csv"
    out_dir = tmp_path / "out_frontier_penalty"
    _frontier_frame().to_csv(data_path, index=False)

    exit_code = run_demo_tune.main(
        [
            "--task",
            "frontier",
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--n-trials",
            "1",
            "--objective",
            "pinball_coverage_penalty",
            "--coverage-target",
            "0.92",
            "--coverage-tolerance",
            "0.02",
            "--penalty-weight",
            "2.5",
        ]
    )
    assert exit_code == 0
    run_dir = sorted([p for p in out_dir.iterdir() if p.is_dir()])[-1]
    used_config = yaml.safe_load((run_dir / "used_config.yaml").read_text(encoding="utf-8"))
    assert used_config["tuning"]["objective"] == "pinball_coverage_penalty"
    assert used_config["tuning"]["coverage_target"] == 0.92
    assert used_config["tuning"]["coverage_tolerance"] == 0.02
    assert used_config["tuning"]["penalty_weight"] == 2.5


def test_run_demo_tune_causal_infers_method_and_balance_threshold(tmp_path) -> None:
    data_path = tmp_path / "causal.csv"
    out_dir = tmp_path / "out_causal"
    frame = _regression_frame()
    frame["treatment"] = (frame["x1"] > frame["x1"].median()).astype(int)
    frame.to_csv(data_path, index=False)

    exit_code = run_demo_tune.main(
        [
            "--task",
            "regression",
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--n-trials",
            "1",
            "--objective",
            "dr_balance_priority",
            "--causal-balance-threshold",
            "0.08",
            "--causal-penalty-weight",
            "3.0",
        ]
    )
    assert exit_code == 0
    run_dir = sorted([p for p in out_dir.iterdir() if p.is_dir()])[-1]
    used_config = yaml.safe_load((run_dir / "used_config.yaml").read_text(encoding="utf-8"))
    assert used_config["causal"]["method"] == "dr"
    assert used_config["causal"]["treatment_col"] == "treatment"
    assert used_config["tuning"]["causal_balance_threshold"] == 0.08
    assert used_config["tuning"]["causal_penalty_weight"] == 3.0


def test_run_demo_tune_causal_default_data_and_study_name_avoid_collision(tmp_path) -> None:
    out_dir = tmp_path / "out_causal_default"
    first = run_demo_tune.main(
        [
            "--task",
            "regression",
            "--objective",
            "dr_balance_priority",
            "--n-trials",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    second = run_demo_tune.main(
        [
            "--task",
            "regression",
            "--objective",
            "dr_balance_priority",
            "--n-trials",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert first == 0
    assert second == 0
    run_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir()])
    assert len(run_dirs) >= 2

    latest_config = yaml.safe_load((run_dirs[-1] / "used_config.yaml").read_text(encoding="utf-8"))
    assert "causal" in latest_config
    assert latest_config["causal"]["method"] == "dr"
    assert latest_config["tuning"]["study_name"].startswith("demo_regression_")
    data_path = latest_config["data"]["path"]
    assert data_path.endswith("causal_dr_tune_demo.csv")

