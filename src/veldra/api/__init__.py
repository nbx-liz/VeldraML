"""Stable public API exports.

This package intentionally uses lazy attribute resolution so importing lightweight
submodules (for example ``veldra.api.exceptions``) does not force heavyweight
runtime dependencies (LightGBM/Optuna/etc.) into memory.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from veldra.api.exceptions import (
    VeldraArtifactError,
    VeldraError,
    VeldraNotImplementedError,
    VeldraValidationError,
)

__all__ = [
    "Artifact",
    "CausalResult",
    "EvalResult",
    "ExportResult",
    "Prediction",
    "RunResult",
    "SimulationResult",
    "TuneResult",
    "VeldraArtifactError",
    "VeldraError",
    "VeldraNotImplementedError",
    "VeldraValidationError",
    "estimate_dr",
    "evaluate",
    "export",
    "fit",
    "predict",
    "simulate",
    "tune",
]


def __getattr__(name: str) -> Any:
    if name == "Artifact":
        return getattr(import_module("veldra.api.artifact"), name)
    if name in {"estimate_dr", "evaluate", "export", "fit", "predict", "simulate", "tune"}:
        return getattr(import_module("veldra.api.runner"), name)
    if name in {
        "CausalResult",
        "EvalResult",
        "ExportResult",
        "Prediction",
        "RunResult",
        "SimulationResult",
        "TuneResult",
    }:
        return getattr(import_module("veldra.api.types"), name)
    raise AttributeError(f"module 'veldra.api' has no attribute '{name}'")
