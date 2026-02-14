"""Config package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CausalConfig",
    "DataConfig",
    "ExportConfig",
    "OnnxOptimizationConfig",
    "PostprocessConfig",
    "RunConfig",
    "SimulationConfig",
    "SplitConfig",
    "TaskConfig",
    "TrainConfig",
    "TuningConfig",
    "load_run_config",
    "migrate_run_config_file",
    "migrate_run_config_payload",
    "save_run_config",
    "MigrationResult",
]


def __getattr__(name: str) -> Any:
    if name in {"load_run_config", "save_run_config"}:
        return getattr(import_module("veldra.config.io"), name)
    if name in {
        "MigrationResult",
        "migrate_run_config_file",
        "migrate_run_config_payload",
    }:
        return getattr(import_module("veldra.config.migrate"), name)
    if name in {
        "CausalConfig",
        "DataConfig",
        "ExportConfig",
        "OnnxOptimizationConfig",
        "PostprocessConfig",
        "RunConfig",
        "SimulationConfig",
        "SplitConfig",
        "TaskConfig",
        "TrainConfig",
        "TuningConfig",
    }:
        return getattr(import_module("veldra.config.models"), name)
    raise AttributeError(f"module 'veldra.config' has no attribute '{name}'")
