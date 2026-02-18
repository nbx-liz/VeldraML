"""Shared lazy runtime helpers for GUI adapter modules."""

from __future__ import annotations

from typing import Any


def resolve_artifact_class(*, cached_class: Any) -> Any:
    """Resolve Artifact class lazily for proxy-based access."""
    if cached_class is None:
        from veldra.api.artifact import Artifact as _Artifact

        return _Artifact
    return cached_class


def resolve_runner_function(name: str) -> Any:
    """Resolve a function from ``veldra.api.runner`` lazily by name."""
    from veldra.api import runner as _runner

    return getattr(_runner, name)


def resolve_data_loader(*, current_loader: Any) -> Any:
    """Resolve tabular data loader lazily while allowing test-time monkeypatch overrides."""
    if current_loader is not None:
        return current_loader
    from veldra.data import load_tabular_data as _load_tabular_data

    return _load_tabular_data
