"""Service helpers used by the Dash adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from veldra.api.artifact import Artifact
from veldra.api.exceptions import (
    VeldraArtifactError,
    VeldraNotImplementedError,
    VeldraValidationError,
)
from veldra.api.runner import estimate_dr, evaluate, export, fit, simulate, tune
from veldra.config.models import RunConfig
from veldra.data import load_tabular_data
from veldra.gui.types import ArtifactSummary, GuiRunResult, RunInvocation


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


def _result_to_payload(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        payload = asdict(result)
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
            result = fit(config)
        elif action == "tune":
            config = _resolve_config(invocation)
            result = tune(config)
        elif action == "estimate_dr":
            config = _resolve_config(invocation)
            result = estimate_dr(config)
        elif action == "evaluate":
            data_path = _require(invocation.data_path, "data_path")
            frame = load_tabular_data(data_path)
            if invocation.artifact_path and invocation.artifact_path.strip():
                artifact = Artifact.load(invocation.artifact_path.strip())
                result = evaluate(artifact, frame)
            else:
                config = _resolve_config(invocation)
                result = evaluate(config, frame)
        elif action == "simulate":
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            data_path = _require(invocation.data_path, "data_path")
            scenarios_path = _require(invocation.scenarios_path, "scenarios_path")
            artifact = Artifact.load(artifact_path)
            frame = load_tabular_data(data_path)
            scenarios = _load_scenarios(scenarios_path)
            result = simulate(artifact, frame, scenarios)
        else:
            artifact_path = _require(invocation.artifact_path, "artifact_path")
            artifact = Artifact.load(artifact_path)
            export_format = (invocation.export_format or "python").strip().lower()
            result = export(artifact, format=export_format)

        return GuiRunResult(
            success=True,
            message=f"{action} completed successfully.",
            payload=_result_to_payload(result),
        )
    except Exception as exc:
        return GuiRunResult(success=False, message=normalize_gui_error(exc), payload={})


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
