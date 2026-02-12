from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from veldra.api.types import SimulationResult
from veldra.gui.types import GuiRunResult

_DASH_AVAILABLE = (
    importlib.util.find_spec("dash") is not None
    and importlib.util.find_spec("dash_bootstrap_components") is not None
)
pytestmark = pytest.mark.skipif(not _DASH_AVAILABLE, reason="Dash GUI dependencies are optional.")


def test_render_page_routes() -> None:
    from veldra.gui.app import render_page

    assert render_page("/config") is not None
    assert render_page("/run") is not None
    assert render_page("/artifacts") is not None
    assert render_page("/unknown") is not None


def test_handle_config_action_paths(monkeypatch) -> None:
    from veldra.gui.app import handle_config_action

    monkeypatch.setattr("veldra.gui.app.load_config_yaml", lambda p: f"loaded:{p}")
    monkeypatch.setattr("veldra.gui.app.save_config_yaml", lambda p, y: f"saved:{p}:{len(y)}")

    class _Cfg:
        class _Task:
            type = "regression"

        task = _Task()

        class _Data:
            target = "y"

        data = _Data()

    monkeypatch.setattr("veldra.gui.app.validate_config", lambda _y: _Cfg())

    loaded, msg1 = handle_config_action("config-load-btn", "x", "a.yaml")
    saved, msg2 = handle_config_action("config-save-btn", "abc", "b.yaml")
    valid, msg3 = handle_config_action("config-validate-btn", "cfg", "c.yaml")

    assert loaded == "loaded:a.yaml"
    assert "Loaded: a.yaml" in msg1
    assert saved == "abc"
    assert "Saved: saved:b.yaml:3" in msg2
    assert valid == "cfg"
    assert "Valid RunConfig: task=regression, target=y" in msg3


def test_handle_config_action_error(monkeypatch) -> None:
    from veldra.gui.app import handle_config_action

    monkeypatch.setattr(
        "veldra.gui.app.validate_config",
        lambda _y: (_ for _ in ()).throw(ValueError("bad")),
    )
    yaml_text, message = handle_config_action("config-validate-btn", "cfg", "c.yaml")
    assert yaml_text == "cfg"
    assert "bad" in message


def test_format_run_action_result(monkeypatch) -> None:
    from veldra.gui.app import format_run_action_result

    monkeypatch.setattr(
        "veldra.gui.app.run_action",
        lambda _inv: GuiRunResult(success=True, message="ok", payload={"a": 1}),
    )
    payload, message = format_run_action_result("fit", "", "", "", "", "", "python")
    assert '"a": 1' in payload
    assert message == "[SUCCESS] ok"

    monkeypatch.setattr(
        "veldra.gui.app.run_action",
        lambda _inv: GuiRunResult(success=False, message="no", payload={}),
    )
    payload2, message2 = format_run_action_result("fit", "", "", "", "", "", "python")
    assert payload2 == "{}"
    assert message2 == "[ERROR] no"


def test_build_artifact_options(monkeypatch) -> None:
    from veldra.gui.app import build_artifact_options
    from veldra.gui.types import ArtifactSummary

    monkeypatch.setattr(
        "veldra.gui.app.list_artifacts",
        lambda _r: [
            ArtifactSummary(
                path="p1",
                run_id="r1",
                task_type="binary",
                created_at_utc="2026-01-01",
            ),
            ArtifactSummary(path="p2", run_id="r2", task_type="regression", created_at_utc=None),
        ],
    )
    options, value = build_artifact_options("artifacts")
    assert len(options) == 2
    assert value == "p1"


def test_format_artifact_metrics(monkeypatch) -> None:
    from veldra.gui.app import format_artifact_metrics

    empty = format_artifact_metrics(None)
    assert "No artifact selected" in str(getattr(empty, "children", ""))

    class _Manifest:
        run_id = "r1"

    class _Task:
        type = "regression"

    class _RunConfig:
        task = _Task()

    class _Artifact:
        manifest = _Manifest()
        run_config = _RunConfig()
        metrics = {"mean": {"rmse": 1.2}}

    monkeypatch.setattr("veldra.gui.app.Artifact.load", lambda _p: _Artifact())
    text = getattr(format_artifact_metrics("a"), "children", "")
    assert "r1" in text
    assert "rmse" in text

    monkeypatch.setattr(
        "veldra.gui.app.Artifact.load",
        lambda _p: (_ for _ in ()).throw(RuntimeError("artifact error")),
    )
    err = getattr(format_artifact_metrics("a"), "children", "")
    assert "artifact error" in err


def test_evaluate_selected_artifact(monkeypatch) -> None:
    from veldra.gui.app import evaluate_selected_artifact

    monkeypatch.setattr("veldra.gui.app.Artifact.load", lambda _p: object())
    monkeypatch.setattr("veldra.gui.app.load_tabular_data", lambda _p: pd.DataFrame({"x": [1, 2]}))
    monkeypatch.setattr(
        "veldra.gui.app.evaluate",
        lambda _a, _f: SimulationResult(task_type="regression", data=pd.DataFrame({"z": [3, 4]})),
    )
    payload = evaluate_selected_artifact("artifact", "data.csv")
    assert "regression" in payload
    assert "z" in payload

    monkeypatch.setattr(
        "veldra.gui.app.Artifact.load",
        lambda _p: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    assert "load failed" in evaluate_selected_artifact("artifact", "data.csv")


def test_handle_config_migrate_helpers(monkeypatch) -> None:
    from veldra.gui.app import handle_config_migrate_apply, handle_config_migrate_preview

    monkeypatch.setattr(
        "veldra.gui.app.migrate_config_from_yaml",
        lambda _yaml, target_version: (
            "normalized: true\n",
            "--- d\n",
            {"target_version": target_version},
        ),
    )
    normalized, diff_text, result_text = handle_config_migrate_preview("a: 1\n", 1)
    assert "normalized" in normalized
    assert "--- d" in diff_text
    assert '"target_version": 1' in result_text

    monkeypatch.setattr(
        "veldra.gui.app.migrate_config_file_via_gui",
        lambda **_: {"output_path": "x.yaml", "changed": True},
    )
    result_apply = handle_config_migrate_apply("in.yaml", "out.yaml", 1)
    assert "x.yaml" in result_apply


def test_build_job_options_and_detail(monkeypatch) -> None:
    from veldra.gui.app import build_job_options, format_job_detail
    from veldra.gui.types import GuiJobRecord, GuiRunResult, RunInvocation

    record = GuiJobRecord(
        job_id="job_1",
        status="queued",
        action="fit",
        created_at_utc="2026-01-01T00:00:00+00:00",
        updated_at_utc="2026-01-01T00:00:00+00:00",
        invocation=RunInvocation(action="fit"),
        result=GuiRunResult(success=True, message="ok", payload={"a": 1}),
    )

    monkeypatch.setattr("veldra.gui.app.list_run_jobs", lambda limit=50: [record])
    options, value, table_text = build_job_options(limit=10)
    assert options[0]["value"] == "job_1"
    assert value == "job_1"
    assert '"job_id": "job_1"' in table_text

    monkeypatch.setattr("veldra.gui.app.get_run_job", lambda _job_id: record)
    detail = format_job_detail("job_1")
    assert '"status": "queued"' in detail
    assert format_job_detail(None) == "No job selected."
    monkeypatch.setattr("veldra.gui.app.get_run_job", lambda _job_id: None)
    assert format_job_detail("missing").startswith("Job not found:")


def test_enqueue_run_job_result(monkeypatch) -> None:
    from veldra.gui.app import enqueue_run_job_result
    from veldra.gui.types import GuiJobResult

    monkeypatch.setattr(
        "veldra.gui.app.submit_run_job",
        lambda _inv: GuiJobResult(job_id="j1", status="queued", message="Queued fit job."),
    )
    payload, message = enqueue_run_job_result("fit", "cfg", "", "", "", "", "python")
    assert '"job_id": "j1"' in payload
    assert message.startswith("[QUEUED]")
