"""Service helpers used by the Dash adapter."""

from __future__ import annotations

import difflib
import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

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
    GuiJobPriority,
    GuiJobRecord,
    GuiJobResult,
    GuiRunResult,
    RunInvocation,
)

LOGGER = logging.getLogger("veldra.gui.services")
_RUNTIME_LOCK = threading.Lock()
_JOB_STORE: GuiJobStore | None = None
_JOB_WORKER: Any | None = None
_JST = timezone(timedelta(hours=9))

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
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        column_profiles: list[dict[str, Any]] = []
        warnings: list[str] = []
        for col in df.columns:
            series = df[col]
            missing_rate = float(series.isna().mean())
            unique_count = int(series.nunique(dropna=True))
            dtype_name = str(series.dtype)
            inferred_kind = (
                "datetime"
                if col in datetime_cols
                else ("numeric" if col in numeric_cols else "categorical")
            )
            col_info = {
                "name": str(col),
                "dtype": dtype_name,
                "kind": inferred_kind,
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "constant": unique_count <= 1,
                "high_missing": missing_rate > 0.5,
                "high_cardinality": inferred_kind == "categorical" and unique_count > 100,
            }
            column_profiles.append(col_info)
            if col_info["high_missing"]:
                warnings.append(f"High missing rate: {col} ({missing_rate:.1%})")
            if col_info["constant"]:
                warnings.append(f"Constant column: {col}")
            if col_info["high_cardinality"]:
                warnings.append(f"High cardinality: {col} ({unique_count})")

        stats = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "datetime_cols": datetime_cols,
            "missing_count": int(df.isnull().sum().sum()),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "column_profiles": column_profiles,
            "warnings": warnings,
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
        normalized, _migration = migrate_run_config_payload(payload, target_version=target_version)
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


def infer_task_type(data: pd.DataFrame, target_col: str) -> str:
    """Infer recommended task type from target column characteristics."""
    if target_col not in data.columns:
        return "regression"
    series = data[target_col].dropna()
    if series.empty:
        return "regression"
    n_unique = int(series.nunique())
    if n_unique == 2:
        return "binary"
    is_int_like = pd.api.types.is_integer_dtype(series)
    if 3 <= n_unique <= 20 and is_int_like:
        return "multiclass"
    return "regression"


@dataclass(slots=True)
class GuardRailResult:
    level: Literal["error", "warning", "info", "ok"]
    message: str
    suggestion: str | None = None


class GuardRailChecker:
    """Best-effort pre-run diagnostics for GUI pages."""

    def check_target(
        self,
        data: pd.DataFrame,
        target_col: str | None,
        task_type: str | None,
        *,
        exclude_cols: list[str] | None = None,
    ) -> list[GuardRailResult]:
        results: list[GuardRailResult] = []
        if not target_col:
            return [GuardRailResult("error", "Target column is required.")]
        if target_col not in data.columns:
            return [GuardRailResult("error", f"Target column not found: {target_col}")]

        exclude_cols = exclude_cols or []
        if target_col in exclude_cols:
            results.append(
                GuardRailResult(
                    "error",
                    "Target column is included in exclude columns.",
                    "Remove target from exclude list.",
                )
            )

        tgt = data[target_col]
        null_rate = float(tgt.isna().mean())
        if null_rate > 0.05:
            results.append(
                GuardRailResult(
                    "warning",
                    f"Target has missing values: {null_rate:.1%}.",
                    "Clean missing labels before training.",
                )
            )

        unique_count = int(tgt.dropna().nunique())
        if task_type == "binary" and unique_count != 2:
            results.append(
                GuardRailResult(
                    "error",
                    f"Binary task selected but unique target values = {unique_count}.",
                    "Use multiclass or adjust target encoding.",
                )
            )
        if task_type == "multiclass" and unique_count > 50:
            results.append(
                GuardRailResult(
                    "warning",
                    f"Multiclass with many classes ({unique_count}).",
                    "Consider class grouping or feature improvements.",
                )
            )

        if task_type in {"binary", "multiclass"}:
            freq = tgt.value_counts(normalize=True, dropna=True)
            if not freq.empty and float(freq.min()) < 0.05:
                results.append(
                    GuardRailResult(
                        "warning",
                        "Class imbalance detected (<5% minority class).",
                        "Enable auto class weight.",
                    )
                )

        if not results:
            results.append(GuardRailResult("ok", "Target checks passed."))
        return results

    def check_validation(
        self,
        data: pd.DataFrame,
        split_config: dict[str, Any],
        *,
        task_type: str | None = None,
        exclude_cols: list[str] | None = None,
    ) -> list[GuardRailResult]:
        results: list[GuardRailResult] = []
        split_type = str(split_config.get("type", "kfold"))
        n_splits = int(split_config.get("n_splits", 5))
        exclude_cols = exclude_cols or []

        if split_type == "timeseries" and not split_config.get("time_col"):
            results.append(
                GuardRailResult(
                    "error",
                    "Time Series split requires a time column.",
                    "Select a time column in Validation.",
                )
            )

        if split_type == "group" and not split_config.get("group_col"):
            results.append(
                GuardRailResult(
                    "error",
                    "Group split requires a group column.",
                    "Select a group column in Validation.",
                )
            )

        if n_splits > max(2, len(data) // 10):
            results.append(
                GuardRailResult(
                    "warning",
                    f"n_splits={n_splits} may be too high for {len(data)} rows.",
                    "Lower fold count or provide more data.",
                )
            )

        if split_type != "timeseries":
            for col in data.columns:
                if col in exclude_cols:
                    continue
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    results.append(
                        GuardRailResult(
                            "warning",
                            f"Datetime feature '{col}' may leak future information.",
                            "Use timeseries split or exclude datetime features.",
                        )
                    )
                    break

        if task_type in {"binary", "multiclass"} and split_type == "kfold":
            results.append(
                GuardRailResult(
                    "info",
                    "Stratified split is usually better for classification.",
                    "Switch split type to stratified when possible.",
                )
            )

        if not results:
            results.append(GuardRailResult("ok", "Validation checks passed."))
        return results

    def check_train(self, config: dict[str, Any]) -> list[GuardRailResult]:
        results: list[GuardRailResult] = []
        lr = float(config.get("learning_rate", 0.05))
        rounds = int(config.get("num_boost_round", 300))
        if lr > 0.3:
            results.append(
                GuardRailResult(
                    "warning",
                    f"Learning rate is high ({lr}).",
                    "Consider <= 0.1 for stable training.",
                )
            )
        if rounds > 5000:
            results.append(
                GuardRailResult(
                    "warning",
                    f"num_boost_round is large ({rounds}).",
                    "Verify early stopping settings.",
                )
            )
        if not results:
            results.append(GuardRailResult("ok", "Train checks passed."))
        return results

    def check_pre_run(self, config_yaml: str, data_path: str | None) -> list[GuardRailResult]:
        results: list[GuardRailResult] = []
        if not data_path or not Path(data_path).exists():
            results.append(
                GuardRailResult(
                    "error",
                    f"Data file not found: {data_path or 'None'}",
                    "Select a valid dataset path.",
                )
            )
        try:
            validate_config(config_yaml)
        except Exception as exc:
            results.append(GuardRailResult("error", f"Config validation error: {exc}"))
        if not results:
            results.append(GuardRailResult("ok", "Pre-run checks passed."))
        return results


def _flatten_numeric_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    top = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if top:
        return top
    mean_obj = metrics.get("mean")
    if isinstance(mean_obj, dict):
        return {k: float(v) for k, v in mean_obj.items() if isinstance(v, (int, float))}
    return {}


def _export_output_path(artifact_path: str, suffix: str) -> Path:
    parent = Path(artifact_path).resolve()
    out_dir = parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).astimezone(_JST).strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{suffix}_{ts}"


def export_excel_report(artifact_path: str) -> str:
    artifact = Artifact.load(_require(artifact_path, "artifact_path"))
    output_path = _export_output_path(artifact_path, "report").with_suffix(".xlsx")

    try:
        from openpyxl import Workbook
    except Exception as exc:
        raise VeldraValidationError(f"Excel export requires openpyxl: {exc}")

    workbook = Workbook()
    sheet_metrics = workbook.active
    sheet_metrics.title = "metrics"
    sheet_metrics.append(["metric", "value"])
    metrics = _flatten_numeric_metrics(getattr(artifact, "metrics", None) or {})
    for key, value in sorted(metrics.items()):
        sheet_metrics.append([key, value])

    sheet_cfg = workbook.create_sheet("config")
    cfg_text = yaml.safe_dump(_to_safe_dict(getattr(artifact, "config", {})), sort_keys=False)
    for idx, line in enumerate(cfg_text.splitlines(), start=1):
        sheet_cfg.cell(row=idx, column=1, value=line)

    sheet_shap = workbook.create_sheet("shap")
    try:
        import shap  # noqa: F401

        sheet_shap.append(["status", "SHAP generation deferred (not computed in GUI MVP)."])
    except Exception:
        sheet_shap.append(["status", "SHAP dependency not installed; sheet skipped."])

    workbook.save(output_path)
    return str(output_path)


def export_html_report(artifact_path: str) -> str:
    artifact = Artifact.load(_require(artifact_path, "artifact_path"))
    output_path = _export_output_path(artifact_path, "report").with_suffix(".html")
    metrics = _flatten_numeric_metrics(getattr(artifact, "metrics", None) or {})
    cfg_text = yaml.safe_dump(_to_safe_dict(getattr(artifact, "config", {})), sort_keys=False)
    task_type = getattr(artifact, "task_type", "unknown")
    run_id = getattr(artifact, "run_id", "unknown")

    rows = "".join(f"<tr><td>{k}</td><td>{v:.6g}</td></tr>" for k, v in sorted(metrics.items()))
    html_text = (
        "<html><head><meta charset='utf-8'><title>Veldra Report</title></head><body>"
        f"<h1>Veldra Report</h1><p>Run ID: {run_id} | Task: {task_type}</p>"
        "<h2>Metrics</h2><table border='1' cellpadding='4'>"
        "<tr><th>Metric</th><th>Value</th></tr>"
        f"{rows}</table>"
        "<h2>Config</h2><pre>"
        f"{cfg_text}"
        "</pre>"
        "</body></html>"
    )
    try:
        from jinja2 import Template

        tmpl = Template(html_text)
        html_text = tmpl.render()
    except Exception:
        pass
    output_path.write_text(html_text, encoding="utf-8")
    return str(output_path)


def _to_safe_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if isinstance(value, dict):
        return value
    return {"value": repr(value)}

@dataclass(slots=True)
class _RunActionContext:
    job_id: str | None = None
    job_store: GuiJobStore | None = None
    action: str | None = None
    tune_total_trials: int | None = None

    def update_progress(self, pct: float, step: str) -> None:
        if not self.job_id or self.job_store is None:
            return
        self.job_store.update_progress(self.job_id, pct, step=step)

    def append_log(
        self,
        *,
        level: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.job_id or self.job_store is None:
            return
        self.job_store.append_job_log(
            self.job_id,
            level=level,
            message=message,
            payload=payload,
        )

    def on_runner_payload(self, level: str, message: str, payload: dict[str, Any]) -> None:
        self.append_log(level=level, message=message, payload=payload)
        if (
            self.action == "tune"
            and self.tune_total_trials is not None
            and self.tune_total_trials > 0
            and "n_trials_done" in payload
        ):
            done = int(payload.get("n_trials_done", 0))
            total = int(self.tune_total_trials)
            pct = 20.0 + (70.0 * max(0, min(done, total)) / float(total))
            self.update_progress(pct, f"tuning_trials_{done}/{total}")


class _RunnerLogCapture(logging.Handler):
    def __init__(self, context: _RunActionContext) -> None:
        super().__init__()
        self._context = context

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {}
        raw = record.getMessage()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {"raw": raw}
        message = str(getattr(record, "event_message", "runner_event"))
        self._context.on_runner_payload(record.levelname, message, payload)


@contextmanager
def _capture_runner_logs(context: _RunActionContext):
    logger = logging.getLogger("veldra")
    handler = _RunnerLogCapture(context)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def run_action(
    invocation: RunInvocation,
    *,
    job_id: str | None = None,
    job_store: GuiJobStore | None = None,
) -> GuiRunResult:
    try:
        action = invocation.action.strip().lower()
        context = _RunActionContext(
            job_id=job_id,
            job_store=job_store,
            action=action,
        )
        context.update_progress(5.0, "validating")
        context.append_log(level="INFO", message="action_started", payload={"action": action})
        if action not in {
            "fit",
            "evaluate",
            "tune",
            "simulate",
            "export",
            "estimate_dr",
            "export_excel",
            "export_html_report",
        }:
            raise VeldraValidationError(f"Unsupported action '{invocation.action}'.")

        if action == "fit":
            context.update_progress(12.0, "load_config")
            config = _resolve_config(invocation)
            context.update_progress(20.0, "fit_running")
            with _capture_runner_logs(context):
                result = _get_runner_func("fit")(config)
            context.update_progress(95.0, "fit_finalize")
        elif action == "tune":
            context.update_progress(12.0, "load_config")
            config = _resolve_config(invocation)
            context.tune_total_trials = int(config.tuning.n_trials)
            context.update_progress(20.0, "tune_running")
            with _capture_runner_logs(context):
                result = _get_runner_func("tune")(config)
            context.update_progress(95.0, "tune_finalize")
        elif action == "estimate_dr":
            context.update_progress(12.0, "load_config")
            config = _resolve_config(invocation)
            context.update_progress(20.0, "estimate_running")
            with _capture_runner_logs(context):
                result = _get_runner_func("estimate_dr")(config)
            context.update_progress(95.0, "estimate_finalize")
        elif action == "evaluate":
            context.update_progress(12.0, "load_data")
            data_path = _require(invocation.data_path, "data_path")
            frame = _get_load_tabular_data()(data_path)
            if invocation.artifact_path and invocation.artifact_path.strip():
                context.update_progress(30.0, "load_artifact")
                artifact = Artifact.load(invocation.artifact_path.strip())
                context.update_progress(45.0, "evaluate_running")
                with _capture_runner_logs(context):
                    result = _get_runner_func("evaluate")(artifact, frame)
            else:
                context.update_progress(30.0, "load_config")
                config = _resolve_config(invocation)
                context.update_progress(45.0, "evaluate_running")
                with _capture_runner_logs(context):
                    result = _get_runner_func("evaluate")(config, frame)
            context.update_progress(95.0, "evaluate_finalize")
        elif action == "simulate":
            context.update_progress(12.0, "load_inputs")
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            data_path = _require(invocation.data_path, "data_path")
            scenarios_path = _require(invocation.scenarios_path, "scenarios_path")
            artifact = Artifact.load(artifact_path)
            frame = _get_load_tabular_data()(data_path)
            scenarios = _load_scenarios(scenarios_path)
            context.update_progress(45.0, "simulate_running")
            with _capture_runner_logs(context):
                result = _get_runner_func("simulate")(artifact, frame, scenarios)
            context.update_progress(95.0, "simulate_finalize")
        elif action == "export_excel":
            context.update_progress(15.0, "load_artifact")
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            context.update_progress(55.0, "export_excel_running")
            output_path = export_excel_report(artifact_path)
            result = {"artifact_path": artifact_path, "output_path": output_path}
            context.update_progress(95.0, "export_excel_finalize")
        elif action == "export_html_report":
            context.update_progress(15.0, "load_artifact")
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            context.update_progress(55.0, "export_html_running")
            output_path = export_html_report(artifact_path)
            result = {"artifact_path": artifact_path, "output_path": output_path}
            context.update_progress(95.0, "export_html_finalize")
        else:
            context.update_progress(15.0, "load_artifact")
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            artifact = Artifact.load(artifact_path)
            export_format = (invocation.export_format or "python").strip().lower()
            context.update_progress(55.0, "export_running")
            with _capture_runner_logs(context):
                result = _get_runner_func("export")(artifact, format=export_format)
            context.update_progress(95.0, "export_finalize")

        context.update_progress(100.0, "completed")
        context.append_log(level="INFO", message="action_completed", payload={"action": action})
        return GuiRunResult(
            success=True,
            message=f"{action} completed successfully.",
            payload=_result_to_payload(result),
        )
    except Exception as exc:
        if job_id and job_store is not None:
            context = _RunActionContext(
                job_id=job_id,
                job_store=job_store,
                action=invocation.action,
            )
            context.append_log(
                level="ERROR",
                message="action_failed",
                payload={"error": normalize_gui_error(exc)},
            )
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


def list_run_job_logs(
    job_id: str,
    *,
    limit: int = 200,
    after_seq: int | None = None,
) -> list[dict[str, Any]]:
    records = get_gui_job_store().list_job_logs(
        _require(job_id, "job_id"),
        limit=limit,
        after_seq=after_seq,
    )
    return [
        {
            "job_id": row.job_id,
            "seq": row.seq,
            "created_at_utc": row.created_at_utc,
            "level": row.level,
            "message": row.message,
            "payload": row.payload,
        }
        for row in records
    ]


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


def set_run_job_priority(job_id: str, priority: str | GuiJobPriority) -> GuiJobResult:
    normalized = str(priority).strip().lower()
    if normalized not in {"low", "normal", "high"}:
        raise VeldraValidationError(f"Unsupported priority: {priority}")
    updated = get_gui_job_store().set_job_priority(
        _require(job_id, "job_id"),
        normalized,
    )
    if updated is None:
        raise VeldraValidationError(f"Job not found: {job_id}")
    log_event(
        LOGGER,
        logging.INFO,
        "gui job priority updated",
        run_id=updated.job_id,
        artifact_path=updated.invocation.artifact_path,
        task_type=updated.action,
        event="gui job priority updated",
        status=updated.status,
        action=updated.action,
        priority=updated.priority,
    )
    return GuiJobResult(
        job_id=updated.job_id,
        status=updated.status,
        message=f"Job {updated.job_id} priority updated to {updated.priority}.",
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


def list_run_jobs_filtered(
    *,
    limit: int = 200,
    status: str | None = None,
    action: str | None = None,
    query: str | None = None,
) -> list[GuiJobRecord]:
    jobs = list_run_jobs(limit=limit, status=(status or None))
    filtered = jobs
    if action:
        action_norm = action.strip().lower()
        filtered = [job for job in filtered if job.action == action_norm]
    if query:
        q = query.strip().lower()
        filtered = [
            job
            for job in filtered
            if q in job.job_id.lower()
            or q in (job.invocation.artifact_path or "").lower()
            or q in (job.invocation.config_path or "").lower()
        ]
    return filtered


def delete_run_jobs(job_ids: list[str]) -> int:
    return get_gui_job_store().delete_jobs(job_ids)


def load_job_config_yaml(job: GuiJobRecord) -> str:
    if job.invocation.config_yaml and job.invocation.config_yaml.strip():
        return job.invocation.config_yaml
    if job.invocation.config_path and job.invocation.config_path.strip():
        return load_config_yaml(job.invocation.config_path)
    raise VeldraValidationError(f"No config source available for job: {job.job_id}")


def compare_artifacts(artifact_a: str, artifact_b: str) -> dict[str, Any]:
    art_a = Artifact.load(_require(artifact_a, "artifact_a"))
    art_b = Artifact.load(_require(artifact_b, "artifact_b"))

    task_a = str(getattr(art_a, "task_type", "unknown"))
    task_b = str(getattr(art_b, "task_type", "unknown"))
    checks: list[dict[str, str]] = []
    if task_a == task_b:
        checks.append({"level": "ok", "message": f"Same task type: {task_a}"})
    else:
        checks.append(
            {
                "level": "warning",
                "message": f"Different task types: {task_a} vs {task_b}",
            }
        )

    cfg_a = _to_safe_dict(getattr(art_a, "config", {}))
    cfg_b = _to_safe_dict(getattr(art_b, "config", {}))
    data_a = ((cfg_a.get("data") or {}) if isinstance(cfg_a, dict) else {}).get("path")
    data_b = ((cfg_b.get("data") or {}) if isinstance(cfg_b, dict) else {}).get("path")
    if data_a and data_b and data_a != data_b:
        checks.append(
            {
                "level": "warning",
                "message": "Different data sources. Row-level prediction diff is disabled.",
            }
        )
    else:
        checks.append({"level": "ok", "message": "Data source appears consistent."})

    split_a = ((cfg_a.get("split") or {}) if isinstance(cfg_a, dict) else {}).get("type")
    split_b = ((cfg_b.get("split") or {}) if isinstance(cfg_b, dict) else {}).get("type")
    if split_a and split_b and split_a != split_b:
        checks.append(
            {
                "level": "info",
                "message": f"Split differs: {split_a} vs {split_b}",
            }
        )

    metrics_a = _flatten_numeric_metrics(getattr(art_a, "metrics", {}) or {})
    metrics_b = _flatten_numeric_metrics(getattr(art_b, "metrics", {}) or {})
    keys = sorted(set(metrics_a) | set(metrics_b))
    metric_rows = []
    for key in keys:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        delta = None
        if va is not None and vb is not None:
            delta = va - vb
        metric_rows.append(
            {
                "metric": key,
                "run_a": va,
                "run_b": vb,
                "delta": delta,
            }
        )

    yaml_a = yaml.safe_dump(cfg_a, sort_keys=False)
    yaml_b = yaml.safe_dump(cfg_b, sort_keys=False)
    return {
        "checks": checks,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "metric_rows": metric_rows,
        "config_yaml_a": yaml_a,
        "config_yaml_b": yaml_b,
    }
