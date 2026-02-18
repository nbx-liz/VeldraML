from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from veldra.api.exceptions import (
    VeldraArtifactError,
    VeldraNotImplementedError,
    VeldraValidationError,
)
from veldra.gui import services
from veldra.gui.types import RunInvocation


def test_default_job_db_path_env(monkeypatch) -> None:
    monkeypatch.setenv("VELDRA_GUI_JOB_DB_PATH", "/tmp/my_jobs.sqlite3")
    assert str(services.default_job_db_path()) == "/tmp/my_jobs.sqlite3"

    monkeypatch.delenv("VELDRA_GUI_JOB_DB_PATH", raising=False)
    assert str(services.default_job_db_path()) == ".veldra_gui/jobs.sqlite3"


def test_lazy_runner_func_and_loader_cache(monkeypatch) -> None:
    import veldra.api.runner as runner_mod
    import veldra.data as data_mod

    original_fit = runner_mod.fit
    original_loader = data_mod.load_tabular_data
    original_cached_fit = services.fit
    original_cached_loader = services.load_tabular_data
    try:
        sentinel_fit = lambda cfg: cfg  # noqa: E731
        runner_mod.fit = sentinel_fit
        services.fit = None
        first = services._get_runner_func("fit")
        assert first is sentinel_fit

        runner_mod.fit = lambda cfg: "changed"  # noqa: E731
        second = services._get_runner_func("fit")
        assert second is sentinel_fit

        sentinel_loader = lambda path: pd.DataFrame({"x": [path]})  # noqa: E731
        data_mod.load_tabular_data = sentinel_loader
        services.load_tabular_data = None
        load1 = services._get_load_tabular_data()
        assert load1 is sentinel_loader

        data_mod.load_tabular_data = lambda path: pd.DataFrame()  # noqa: E731
        load2 = services._get_load_tabular_data()
        assert load2 is sentinel_loader
    finally:
        runner_mod.fit = original_fit
        data_mod.load_tabular_data = original_loader
        services.fit = original_cached_fit
        services.load_tabular_data = original_cached_loader


def test_list_artifacts_not_directory_and_broken_manifest(tmp_path) -> None:
    not_dir = tmp_path / "file.txt"
    not_dir.write_text("x", encoding="utf-8")
    with pytest.raises(VeldraValidationError, match="not a directory"):
        services.list_artifacts(str(not_dir))

    root = tmp_path / "arts"
    root.mkdir()
    a = root / "a1"
    a.mkdir()
    (a / "manifest.json").write_text("{bad json", encoding="utf-8")
    items = services.list_artifacts(str(root))
    assert len(items) == 1
    assert items[0].task_type == "unknown"


def test_normalize_gui_error_variants() -> None:
    assert "Validation error" in services.normalize_gui_error(VeldraValidationError("x"))
    assert "Artifact error" in services.normalize_gui_error(VeldraArtifactError("x"))
    assert "Not implemented" in services.normalize_gui_error(VeldraNotImplementedError("x"))
    assert "RuntimeError" in services.normalize_gui_error(RuntimeError("x"))
    assert services.classify_gui_error(VeldraValidationError("x")) == "validation"
    assert services.classify_gui_error(FileNotFoundError("x")) == "file_not_found"
    assert services.classify_gui_error(PermissionError("x")) == "permission"
    assert services.classify_gui_error(TimeoutError("x")) == "timeout"
    assert services.classify_gui_error(RuntimeError("database is locked")) == "resource_busy"


def test_run_action_missing_required_inputs() -> None:
    assert "required" in services.run_action(RunInvocation(action="fit")).message
    assert "required" in services.run_action(RunInvocation(action="tune")).message
    assert "required" in services.run_action(RunInvocation(action="estimate_dr")).message
    assert (
        "required"
        in services.run_action(RunInvocation(action="evaluate", artifact_path="a_only")).message
    )
    assert "required" in services.run_action(RunInvocation(action="simulate")).message
    assert "required" in services.run_action(RunInvocation(action="export")).message


def test_run_action_evaluate_artifact_and_config_paths(monkeypatch) -> None:
    frame = pd.DataFrame({"x": [1.0], "y": [2.0]})
    monkeypatch.setattr(services, "_get_load_tabular_data", lambda: (lambda _path: frame))
    monkeypatch.setattr(services, "Artifact", SimpleNamespace(load=lambda _path: "artifact"))
    monkeypatch.setattr(
        services,
        "_get_runner_func",
        lambda name: (lambda obj, data: {"name": name, "obj": obj, "n_rows": len(data)}),
    )

    # artifact evaluate path
    r1 = services.run_action(
        RunInvocation(action="evaluate", artifact_path="art", data_path="eval.csv")
    )
    assert r1.success is True
    assert r1.payload["result"]["name"] == "evaluate"

    # config evaluate path
    cfg_yaml = (
        "config_version: 1\n"
        "task:\n"
        "  type: regression\n"
        "data:\n"
        "  path: train.csv\n"
        "  target: y\n"
    )
    r2 = services.run_action(RunInvocation(action="evaluate", config_yaml=cfg_yaml, data_path="x"))
    assert r2.success is True
