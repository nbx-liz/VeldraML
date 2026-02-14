"""Service helpers used by the Dash adapter."""

from __future__ import annotations

import difflib
import json
import logging
import os
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from veldra.api.exceptions import (
    VeldraArtifactError,
    VeldraNotImplementedError,
    VeldraValidationError,
)
from veldra.api.logging import log_event
from veldra.config.migrate import migrate_run_config_file, migrate_run_config_payload
from veldra.config.models import RunConfig
from veldra.gui.job_store import GuiJobStore
from veldra.gui.types import (
    ArtifactSummary,
    GuiJobRecord,
    GuiJobResult,
    GuiRunResult,
    RunInvocation,
)

LOGGER = logging.getLogger("veldra.gui.services")
_RUNTIME_LOCK = threading.Lock()
_JOB_STORE: GuiJobStore | None = None
_JOB_WORKER: Any | None = None

# Lazy runtime symbols for heavyweight deps.
_ARTIFACT_CLS: Any | None = None
evaluate: Any | None = None
estimate_dr: Any | None = None
export: Any | None = None
fit: Any | None = None
simulate: Any | None = None
tune: Any | None = None
load_tabular_data: Any | None = None


def _get_artifact_cls() -> Any:
    global Artifact, _ARTIFACT_CLS
    if Artifact is not _ArtifactProxy:
        return Artifact
    if _ARTIFACT_CLS is None:
        from veldra.api.artifact import Artifact as _Artifact

        _ARTIFACT_CLS = _Artifact
    return _ARTIFACT_CLS


class _ArtifactProxy:
    @staticmethod
    def load(path: str) -> Any:
        return _get_artifact_cls().load(path)


# Backward-compat name used in tests via monkeypatch("...Artifact.load", ...).
Artifact: Any = _ArtifactProxy


def _get_runner_func(name: str) -> Any:
    global evaluate, estimate_dr, export, fit, simulate, tune
    current = {
        "evaluate": evaluate,
        "estimate_dr": estimate_dr,
        "export": export,
        "fit": fit,
        "simulate": simulate,
        "tune": tune,
    }.get(name)
    if current is not None:
        return current
    from veldra.api import runner as _runner

    resolved = getattr(_runner, name)
    if name == "evaluate":
        evaluate = resolved
    elif name == "estimate_dr":
        estimate_dr = resolved
    elif name == "export":
        export = resolved
    elif name == "fit":
        fit = resolved
    elif name == "simulate":
        simulate = resolved
    elif name == "tune":
        tune = resolved
    return resolved


def _get_load_tabular_data() -> Any:
    global load_tabular_data
    if load_tabular_data is None:
        from veldra.data import load_tabular_data as _load_tabular_data

        load_tabular_data = _load_tabular_data
    return load_tabular_data


def inspect_data(path: str) -> dict[str, Any]:
    """Inspect a data file and return preview and statistics."""
    try:
        data_path = Path(_require(path, "path"))
        if not data_path.exists():
            raise VeldraValidationError(f"Data file does not exist: {data_path}")
        loader = _get_load_tabular_data()
        df = loader(str(data_path))
        
        # Calculate column stats
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        
        stats = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "missing_count": int(df.isnull().sum().sum()),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
        }

        # Preview data (handle non-serializable types if any)
        preview = df.head(10).astype(object).where(pd.notnull(df), None)
        
        return {
            "success": True,
            "stats": stats,
            "preview": preview.to_dict(orient="records"),
            "path": str(data_path.resolve()),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


def default_job_db_path() -> Path:
    return Path(os.getenv("VELDRA_GUI_JOB_DB_PATH", ".veldra_gui/jobs.sqlite3"))


def set_gui_runtime(*, job_store: GuiJobStore, worker: Any | None) -> None:
    global _JOB_STORE, _JOB_WORKER
    with _RUNTIME_LOCK:
        _JOB_STORE = job_store
        _JOB_WORKER = worker


def get_gui_job_store() -> GuiJobStore:
    global _JOB_STORE
    with _RUNTIME_LOCK:
        if _JOB_STORE is None:
            _JOB_STORE = GuiJobStore(default_job_db_path())
        return _JOB_STORE


def stop_gui_runtime() -> None:
    global _JOB_STORE, _JOB_WORKER
    with _RUNTIME_LOCK:
        worker = _JOB_WORKER
        _JOB_WORKER = None
        _JOB_STORE = None
    if worker is not None and hasattr(worker, "stop"):
        worker.stop()


def _start_worker_if_needed() -> None:
    with _RUNTIME_LOCK:
        worker = _JOB_WORKER
    if worker is not None and hasattr(worker, "start"):
        worker.start()


def normalize_gui_error(exc: Exception) -> str:
    if isinstance(exc, VeldraValidationError):
        return f"Validation error: {exc}"
    if isinstance(exc, VeldraArtifactError):
        return f"Artifact error: {exc}"
    if isinstance(exc, VeldraNotImplementedError):
        return f"Not implemented: {exc}"
    return f"{exc.__class__.__name__}: {exc}"


def _require(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise VeldraValidationError(f"{field_name} is required.")
    return value.strip()


def _load_config_from_yaml(yaml_text: str) -> RunConfig:
    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise VeldraValidationError("Config YAML must deserialize to an object.")
    return RunConfig.model_validate(raw)


def validate_config(yaml_text: str) -> RunConfig:
    return _load_config_from_yaml(yaml_text)


def load_config_yaml(path: str) -> str:
    config_path = Path(_require(path, "config_path"))
    if not config_path.exists():
        raise VeldraValidationError(f"Config file does not exist: {config_path}")
    return config_path.read_text(encoding="utf-8")


def save_config_yaml(path: str, yaml_text: str) -> str:
    config_path = Path(_require(path, "config_path"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml_text, encoding="utf-8")
    return str(config_path)


def migrate_config_from_yaml(
    yaml_text: str,
    target_version: int = 1,
) -> tuple[str, str]:
    """Migrate config from YAML string. Returns (normalized_yaml, diff)."""
    try:
        payload = yaml.safe_load(yaml_text or "")
        if not isinstance(payload, dict):
            raise VeldraValidationError("Config YAML must deserialize to an object.")
        normalized, result = migrate_run_config_payload(payload, target_version=target_version)
        normalized_yaml = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
        
        # Calculate diff
        input_lines = (yaml_text or "").splitlines()
        output_lines = normalized_yaml.splitlines()
        diff = "\n".join(
            difflib.unified_diff(
                input_lines,
                output_lines,
                fromfile="Original",
                tofile="Normalized",
                lineterm="",
            )
        )
        return normalized_yaml, diff
    except Exception as exc:
        raise VeldraValidationError(f"Migration failed: {exc}")


def migrate_config_file_via_gui(
    input_path: str,
    output_path: str | None = None,
    target_version: int = 1,
) -> str:
    """Migrate config file. Returns summary message."""
    try:
        result = migrate_run_config_file(
            input_path=input_path,
            output_path=output_path,
            target_version=target_version,
        )
        return (
            f"Migration successful.\\n"
            f"Source: {result.input_path}\\n"
            f"Output: {result.output_path}\\n"
            f"Version: {result.source_version} -> {result.target_version}\\n"
            f"Changed: {result.changed}"
        )
    except Exception as exc:
        raise VeldraValidationError(f"File migration failed: {exc}")


def _result_to_payload(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        # Handle EvalResult or others that might have DataFrame in fields
        # asdict fails on DataFrame. We need to handle it manually or use asdict safely
        try:
            payload = asdict(result)
        except Exception:
            # Fallback for when asdict fails (e.g. DataFrame in field)
            payload = {}
            for field in result.__dataclass_fields__:
                val = getattr(result, field)
                if isinstance(val, pd.DataFrame):
                    payload[field] = {
                        "n_rows": int(len(val)),
                        "columns": list(val.columns),
                        "preview": val.head(20).to_dict(orient="records"),
                    }
                else:
                     payload[field] = val
    elif isinstance(result, pd.DataFrame):
        payload = {
            "n_rows": int(len(result)),
            "columns": list(result.columns),
            "preview": result.head(20).to_dict(orient="records"),
        }
    elif isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        payload = {"result": result}
    else:
        payload = {"result_repr": repr(result)}

    data_obj = payload.get("data")
    if isinstance(data_obj, pd.DataFrame):
        payload["data"] = {
            "n_rows": int(len(data_obj)),
            "columns": list(data_obj.columns),
            "preview": data_obj.head(20).to_dict(orient="records"),
        }
    return payload


def _resolve_config(invocation: RunInvocation) -> RunConfig:
    if invocation.config_yaml and invocation.config_yaml.strip():
        return _load_config_from_yaml(invocation.config_yaml)
    if invocation.config_path and invocation.config_path.strip():
        config_text = load_config_yaml(invocation.config_path)
        return _load_config_from_yaml(config_text)
    raise VeldraValidationError("Either config YAML or config path is required.")


def _load_scenarios(path: str) -> Any:
    scenarios_path = Path(_require(path, "scenarios_path"))
    if not scenarios_path.exists():
        raise VeldraValidationError(f"Scenarios file does not exist: {scenarios_path}")
    text = scenarios_path.read_text(encoding="utf-8")
    if scenarios_path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def run_action(invocation: RunInvocation) -> GuiRunResult:
    try:
        action = invocation.action.strip().lower()
        if action not in {"fit", "evaluate", "tune", "simulate", "export", "estimate_dr"}:
            raise VeldraValidationError(f"Unsupported action '{invocation.action}'.")

        if action == "fit":
            config = _resolve_config(invocation)
            result = _get_runner_func("fit")(config)
        elif action == "tune":
            config = _resolve_config(invocation)
            result = _get_runner_func("tune")(config)
        elif action == "estimate_dr":
            config = _resolve_config(invocation)
            result = _get_runner_func("estimate_dr")(config)
        elif action == "evaluate":
            data_path = _require(invocation.data_path, "data_path")
            frame = _get_load_tabular_data()(data_path)
            if invocation.artifact_path and invocation.artifact_path.strip():
                artifact = Artifact.load(invocation.artifact_path.strip())
                result = _get_runner_func("evaluate")(artifact, frame)
            else:
                config = _resolve_config(invocation)
                result = _get_runner_func("evaluate")(config, frame)
        elif action == "simulate":
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            data_path = _require(invocation.data_path, "data_path")
            scenarios_path = _require(invocation.scenarios_path, "scenarios_path")
            artifact = Artifact.load(artifact_path)
            frame = _get_load_tabular_data()(data_path)
            scenarios = _load_scenarios(scenarios_path)
            result = _get_runner_func("simulate")(artifact, frame, scenarios)
        else:
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            artifact = Artifact.load(artifact_path)
            export_format = (invocation.export_format or "python").strip().lower()
            result = _get_runner_func("export")(artifact, format=export_format)

        return GuiRunResult(
            success=True,
            message=f"{action} completed successfully.",
            payload=_result_to_payload(result),
        )
    except Exception as exc:
        return GuiRunResult(success=False, message=normalize_gui_error(exc), payload={})


def submit_run_job(invocation: RunInvocation) -> GuiJobResult:
    store = get_gui_job_store()
    job = store.enqueue_job(invocation)
    _start_worker_if_needed()
    log_event(
        LOGGER,
        logging.INFO,
        "gui job submitted",
        run_id=job.job_id,
        artifact_path=invocation.artifact_path,
        task_type=job.action,
        event="gui job submitted",
        status=job.status,
        action=job.action,
    )
    return GuiJobResult(
        job_id=job.job_id,
        status=job.status,
        message=f"Queued {job.action} job.",
    )


def get_run_job(job_id: str) -> GuiJobRecord | None:
    return get_gui_job_store().get_job(_require(job_id, "job_id"))


def list_run_jobs(*, limit: int = 100, status: str | None = None) -> list[GuiJobRecord]:
    return get_gui_job_store().list_jobs(limit=limit, status=status)


def cancel_run_job(job_id: str) -> GuiJobResult:
    canceled = get_gui_job_store().request_cancel(_require(job_id, "job_id"))
    if canceled is None:
        raise VeldraValidationError(f"Job not found: {job_id}")
    log_event(
        LOGGER,
        logging.INFO,
        "gui job updated",
        run_id=canceled.job_id,
        artifact_path=canceled.invocation.artifact_path,
        task_type=canceled.action,
        event="gui job updated",
        status=canceled.status,
        action=canceled.action,
        cancel_requested=canceled.cancel_requested,
    )
    return GuiJobResult(
        job_id=canceled.job_id,
        status=canceled.status,
        message=f"Job {canceled.job_id} status updated to {canceled.status}.",
    )


def list_artifacts(root_dir: str) -> list[ArtifactSummary]:
    root = Path(_require(root_dir, "root_dir"))
    if not root.exists():
        return []
    if not root.is_dir():
        raise VeldraValidationError(f"Artifact root is not a directory: {root}")

    summaries: list[ArtifactSummary] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summaries.append(
                ArtifactSummary(
                    path=str(child),
                    run_id=str(manifest.get("run_id", child.name)),
                    task_type=str(manifest.get("task_type", "unknown")),
                    created_at_utc=(
                        str(manifest["created_at_utc"])
                        if manifest.get("created_at_utc") is not None
                        else None
                    ),
                )
            )
        except Exception:
            summaries.append(
                ArtifactSummary(
                    path=str(child),
                    run_id=child.name,
                    task_type="unknown",
                    created_at_utc=None,
                )
            )

    summaries.sort(key=lambda item: item.created_at_utc or "", reverse=True)
    return summaries
