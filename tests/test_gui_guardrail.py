from __future__ import annotations

import pandas as pd

from veldra.gui.services import GuardRailChecker


def test_guardrail_target_and_validation() -> None:
    checker = GuardRailChecker()
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, None],
            "grp": [1, 1, 2, 2, 3, 3],
            "dt": pd.to_datetime(["2024-01-01"] * 6),
        }
    )

    target_findings = checker.check_target(frame, "target", "binary", exclude_cols=[])
    assert target_findings

    validation_findings = checker.check_validation(
        frame,
        {"type": "group", "n_splits": 10, "group_col": None},
        task_type="binary",
        exclude_cols=[],
    )
    assert any(item.level == "error" for item in validation_findings)


def test_guardrail_train_and_pre_run(tmp_path) -> None:
    checker = GuardRailChecker()
    train_findings = checker.check_train({"learning_rate": 0.5, "num_boost_round": 6000})
    assert any(item.level == "warning" for item in train_findings)

    data_path = tmp_path / "x.csv"
    data_path.write_text("a,target\n1,0\n2,1\n", encoding="utf-8")
    config_yaml = (
        "config_version: 1\n"
        "task:\n  type: binary\n"
        "data:\n  path: x.csv\n  target: target\n"
    )
    pre = checker.check_pre_run(config_yaml, str(data_path))
    assert pre
