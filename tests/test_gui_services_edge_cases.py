from __future__ import annotations

import builtins
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.gui import services
from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation


def test_get_artifact_cls_and_runner_lazy_cache(monkeypatch) -> None:
    original = services.Artifact
    original_cls = services._ARTIFACT_CLS
    try:
        sentinel = object()
        services.Artifact = sentinel
        assert services._get_artifact_cls() is sentinel

        services.Artifact = services._ArtifactProxy
        services._ARTIFACT_CLS = None
        cls1 = services._get_artifact_cls()
        cls2 = services._get_artifact_cls()
        assert cls1 is cls2
    finally:
        services.Artifact = original
        services._ARTIFACT_CLS = original_cls

    for name in ("evaluate", "estimate_dr", "export", "fit", "simulate", "tune"):
        setattr(services, name, None)
        fn = services._get_runner_func(name)
        assert callable(fn)


def test_inspect_data_warning_profiles_and_infer_task_type(monkeypatch, tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x", encoding="utf-8")
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "const": [1, 1, 1, 1],
            "high_missing": [1.0, None, None, None],
            "high_card": [f"c{i}" for i in range(4)],
        }
    )
    monkeypatch.setattr(services, "_get_load_tabular_data", lambda: (lambda _p: df))

    result = services.inspect_data(str(path))
    assert result["success"] is True
    warnings = "\n".join(result["stats"]["warnings"])
    assert "High missing rate" in warnings
    assert "Constant column" in warnings

    assert services.infer_task_type(df, "missing_target") == "regression"
    assert services.infer_task_type(pd.DataFrame({"y": []}), "y") == "regression"
    assert services.infer_task_type(pd.DataFrame({"y": [0, 1]}), "y") == "binary"
    assert services.infer_task_type(pd.DataFrame({"y": [0, 1, 2, 1]}), "y") == "multiclass"
    assert services.infer_task_type(pd.DataFrame({"y": [0.1, 0.2, 0.3]}), "y") == "regression"


def test_guardrail_checker_edge_branches(tmp_path) -> None:
    checker = services.GuardRailChecker()
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, None],
            "dt": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "x": [1, 2, 3, 4],
        }
    )

    miss_target = checker.check_target(frame, None, "binary")
    assert miss_target[0].level == "error"

    not_found = checker.check_target(frame, "zzz", "binary")
    assert not_found[0].level == "error"

    findings = checker.check_target(frame, "target", "binary", exclude_cols=["target"])
    levels = {item.level for item in findings}
    assert "error" in levels
    assert "warning" in levels

    val_findings = checker.check_validation(
        frame,
        {"type": "timeseries", "n_splits": 20, "time_col": None},
        task_type="binary",
        exclude_cols=[],
    )
    assert any(item.level == "error" for item in val_findings)

    ok_validation = checker.check_validation(
        pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}),
        {"type": "kfold", "n_splits": 2},
        task_type="regression",
        exclude_cols=[],
    )
    assert any(item.level == "ok" for item in ok_validation)

    ok_train = checker.check_train({"learning_rate": 0.05, "num_boost_round": 100})
    assert ok_train[0].level == "ok"

    config_yaml = "config_version: 1\ntask:\n  type: regression\ndata:\n  path: p\n  target: y\n"
    pre = checker.check_pre_run(config_yaml, str(tmp_path / "missing.csv"))
    assert any(item.level == "error" for item in pre)


def test_flatten_numeric_metrics_and_safe_dict() -> None:
    assert services._flatten_numeric_metrics(None) == {}
    assert services._flatten_numeric_metrics({"a": 1, "b": 1.2}) == {"a": 1.0, "b": 1.2}
    assert services._flatten_numeric_metrics({"mean": {"x": 1, "z": "n"}}) == {"x": 1.0}
    assert services._flatten_numeric_metrics({"mean": []}) == {}

    @dataclass
    class _DataclassObj:
        x: int

    assert services._to_safe_dict(_DataclassObj(1)) == {"x": 1}

    class _DumpOk:
        def model_dump(self, mode: str = "json"):
            _ = mode
            return {"m": 1}

    class _DumpBad:
        def model_dump(self, mode: str = "json"):
            _ = mode
            raise RuntimeError("boom")

    assert services._to_safe_dict(_DumpOk()) == {"m": 1}
    dumped_bad = services._to_safe_dict(_DumpBad())
    assert isinstance(dumped_bad.get("value"), str)
    assert "_DumpBad object" in dumped_bad["value"]
    assert services._to_safe_dict({"k": 1}) == {"k": 1}


def test_export_report_functions_cover_optional_branches(monkeypatch, tmp_path) -> None:
    artifact_root = tmp_path / "artifacts" / "run1"
    artifact_root.mkdir(parents=True, exist_ok=True)

    fake_art = SimpleNamespace(
        metrics={"mean": {"rmse": 0.25}},
        config={"task": {"type": "regression"}},
        task_type="regression",
        run_id="r1",
    )
    monkeypatch.setattr(services, "Artifact", SimpleNamespace(load=lambda _p: fake_art))

    original_import = builtins.__import__

    class _Sheet:
        def __init__(self) -> None:
            self.title = ""
            self.rows: list[list[object]] = []

        def append(self, row) -> None:
            self.rows.append(list(row))

        def cell(self, row: int, column: int, value: object) -> None:  # noqa: ARG002
            self.rows.append([value])

    class _Workbook:
        def __init__(self) -> None:
            self.active = _Sheet()
            self.sheets = [self.active]

        def create_sheet(self, _name: str) -> _Sheet:
            sheet = _Sheet()
            self.sheets.append(sheet)
            return sheet

        def save(self, path: str | Path) -> None:
            Path(path).write_text("fake-xlsx", encoding="utf-8")

    fake_openpyxl = types.SimpleNamespace(Workbook=_Workbook)

    def _import_with_shap(name, *args, **kwargs):
        if name == "openpyxl":
            return fake_openpyxl
        if name == "shap":
            return types.SimpleNamespace(__name__="shap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_with_shap)
    excel_out = Path(services.export_excel_report(str(artifact_root)))
    assert excel_out.is_file()

    def _import_fail_openpyxl(name, *args, **kwargs):
        if name == "openpyxl":
            raise ImportError("missing openpyxl")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_fail_openpyxl)
    with pytest.raises(VeldraValidationError, match="openpyxl"):
        services.export_excel_report(str(artifact_root))

    monkeypatch.setattr(builtins, "__import__", _import_with_shap)
    html_out = Path(services.export_html_report(str(artifact_root)))
    assert html_out.is_file()

    def _import_fail_jinja(name, *args, **kwargs):
        if name == "jinja2":
            raise ImportError("missing jinja2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_fail_jinja)
    html_out2 = Path(services.export_html_report(str(artifact_root)))
    assert html_out2.is_file()


def test_run_action_export_report_variants(monkeypatch) -> None:
    monkeypatch.setattr(services, "export_excel_report", lambda _p: "out.xlsx")
    monkeypatch.setattr(services, "export_html_report", lambda _p: "out.html")

    r1 = services.run_action(RunInvocation(action="export_excel", artifact_path="artifacts/a"))
    r2 = services.run_action(
        RunInvocation(action="export_html_report", artifact_path="artifacts/a")
    )

    assert r1.success is True
    assert r1.payload["result"]["output_path"] == "out.xlsx"
    assert r2.success is True
    assert r2.payload["result"]["output_path"] == "out.html"


def test_job_services_filter_delete_cancel_and_load_config_yaml(monkeypatch, tmp_path) -> None:
    class _Store:
        def __init__(self) -> None:
            self.deleted = 0

        def request_cancel(self, job_id: str):
            if job_id == "missing":
                return None
            return GuiJobRecord(
                job_id=job_id,
                status="cancel_requested",
                action="fit",
                created_at_utc="2026-01-01T00:00:00+00:00",
                updated_at_utc="2026-01-01T00:00:00+00:00",
                invocation=RunInvocation(action="fit", artifact_path="artifacts/r1"),
                result=GuiRunResult(success=True, message="ok", payload={}),
                cancel_requested=True,
            )

        def delete_jobs(self, job_ids: list[str]) -> int:
            self.deleted += len(job_ids)
            return len(job_ids)

        def set_job_priority(self, job_id: str, priority: str):
            if job_id == "missing":
                return None
            if job_id == "running":
                raise ValueError("Priority can only be changed for queued jobs.")
            return GuiJobRecord(
                job_id=job_id,
                status="queued",
                action="fit",
                created_at_utc="2026-01-01T00:00:00+00:00",
                updated_at_utc="2026-01-01T00:00:00+00:00",
                invocation=RunInvocation(action="fit", artifact_path="artifacts/r1"),
                result=GuiRunResult(success=True, message="ok", payload={}),
                priority=priority,  # type: ignore[arg-type]
            )

    store = _Store()
    services.set_gui_runtime(job_store=store, worker=None)

    with pytest.raises(VeldraValidationError, match="Job not found"):
        services.cancel_run_job("missing")

    canceled = services.cancel_run_job("j1")
    assert canceled.status == "cancel_requested"
    updated = services.set_run_job_priority("j1", "high")
    assert "priority updated" in updated.message

    assert services.delete_run_jobs(["a", "b"]) == 2

    with pytest.raises(VeldraValidationError, match="Job not found"):
        services.set_run_job_priority("missing", "high")

    jobs = [
        GuiJobRecord(
            job_id="job-1",
            status="queued",
            action="fit",
            created_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            invocation=RunInvocation(
                action="fit",
                artifact_path="artifacts/a",
                config_path="cfg_a.yaml",
            ),
        ),
        GuiJobRecord(
            job_id="job-2",
            status="queued",
            action="tune",
            created_at_utc="2026-01-01T00:00:00+00:00",
            updated_at_utc="2026-01-01T00:00:00+00:00",
            invocation=RunInvocation(
                action="tune",
                artifact_path="artifacts/b",
                config_path="cfg_b.yaml",
            ),
        ),
    ]
    monkeypatch.setattr(services, "list_run_jobs", lambda **_k: jobs)
    filtered = services.list_run_jobs_filtered(action="fit", query="artifacts/a")
    assert [j.job_id for j in filtered] == ["job-1"]

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    job_with_yaml = GuiJobRecord(
        job_id="j1",
        status="queued",
        action="fit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="fit", config_yaml="a: 1\n"),
    )
    assert services.load_job_config_yaml(job_with_yaml).strip() == "a: 1"

    job_with_path = GuiJobRecord(
        job_id="j2",
        status="queued",
        action="fit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="fit", config_path=str(cfg)),
    )
    assert services.load_job_config_yaml(job_with_path).strip() == "x: 1"

    no_cfg = GuiJobRecord(
        job_id="j3",
        status="queued",
        action="fit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="fit"),
    )
    with pytest.raises(VeldraValidationError, match="No config source"):
        services.load_job_config_yaml(no_cfg)


def test_compare_artifacts_branches(monkeypatch) -> None:
    art_a = SimpleNamespace(
        task_type="binary",
        metrics={"auc": 0.8},
        config={"data": {"path": "data/a.csv"}, "split": {"type": "kfold"}},
    )
    art_b = SimpleNamespace(
        task_type="regression",
        metrics={"auc": 0.7, "rmse": 1.0},
        config={"data": {"path": "data/b.csv"}, "split": {"type": "timeseries"}},
    )

    monkeypatch.setattr(
        services,
        "Artifact",
        SimpleNamespace(load=lambda p: art_a if str(p).endswith("a") else art_b),
    )
    payload = services.compare_artifacts("/tmp/a", "/tmp/b")
    assert payload["metric_rows"]
    joined = "\n".join(item["message"] for item in payload["checks"])
    assert "Different task types" in joined
    assert "Different data sources" in joined
    assert "Split differs" in joined
