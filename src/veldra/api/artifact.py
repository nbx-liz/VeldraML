"""Stable Artifact API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veldra import __version__
from veldra.api.exceptions import VeldraNotImplementedError
from veldra.artifact.manifest import Manifest, build_manifest
from veldra.artifact.store import load_artifact, save_artifact
from veldra.config.models import RunConfig


class Artifact:
    """Serializable artifact object passed across API/CLI/GUI boundaries."""

    def __init__(
        self,
        run_config: RunConfig,
        manifest: Manifest,
        feature_schema: dict[str, Any] | None = None,
    ) -> None:
        self.run_config = run_config
        self.manifest = manifest
        self.feature_schema = feature_schema or {}

    @classmethod
    def from_config(
        cls,
        run_config: RunConfig,
        run_id: str,
        feature_schema: dict[str, Any] | None = None,
    ) -> "Artifact":
        manifest = build_manifest(
            run_config=run_config,
            run_id=run_id,
            project_version=__version__,
        )
        return cls(run_config=run_config, manifest=manifest, feature_schema=feature_schema)

    @classmethod
    def load(cls, path: str | Path) -> "Artifact":
        run_config, manifest, feature_schema = load_artifact(path)
        return cls(run_config=run_config, manifest=manifest, feature_schema=feature_schema)

    def save(self, path: str | Path) -> None:
        save_artifact(
            path=path,
            run_config=self.run_config,
            manifest=self.manifest,
            feature_schema=self.feature_schema,
        )

    def predict(self, df: Any) -> Any:
        raise VeldraNotImplementedError(
            "Artifact.predict is not implemented in MVP scaffold."
        )

    def simulate(self, df: Any, scenario: dict[str, Any]) -> Any:
        raise VeldraNotImplementedError(
            "Artifact.simulate is not implemented in MVP scaffold."
        )
