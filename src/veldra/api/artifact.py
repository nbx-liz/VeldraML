"""Stable Artifact API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from veldra import __version__
from veldra.api.exceptions import VeldraNotImplementedError, VeldraValidationError
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
        model_text: str | None = None,
        metrics: dict[str, Any] | None = None,
        cv_results: pd.DataFrame | None = None,
    ) -> None:
        self.run_config = run_config
        self.manifest = manifest
        self.feature_schema = feature_schema or {}
        self.model_text = model_text
        self.metrics = metrics or {}
        self.cv_results = cv_results
        self._booster: lgb.Booster | None = None

    @classmethod
    def from_config(
        cls,
        run_config: RunConfig,
        run_id: str,
        feature_schema: dict[str, Any] | None = None,
        model_text: str | None = None,
        metrics: dict[str, Any] | None = None,
        cv_results: pd.DataFrame | None = None,
    ) -> "Artifact":
        manifest = build_manifest(
            run_config=run_config,
            run_id=run_id,
            project_version=__version__,
        )
        return cls(
            run_config=run_config,
            manifest=manifest,
            feature_schema=feature_schema,
            model_text=model_text,
            metrics=metrics,
            cv_results=cv_results,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Artifact":
        run_config, manifest, feature_schema, extras = load_artifact(path)
        return cls(
            run_config=run_config,
            manifest=manifest,
            feature_schema=feature_schema,
            model_text=extras.get("model_text"),
            metrics=extras.get("metrics"),
            cv_results=extras.get("cv_results"),
        )

    def save(self, path: str | Path) -> None:
        save_artifact(
            path=path,
            run_config=self.run_config,
            manifest=self.manifest,
            feature_schema=self.feature_schema,
            model_text=self.model_text,
            metrics=self.metrics,
            cv_results=self.cv_results,
        )

    def _get_booster(self) -> lgb.Booster:
        if self.model_text is None:
            raise VeldraValidationError("Artifact model is missing. Run fit before predict.")
        if self._booster is None:
            self._booster = lgb.Booster(model_str=self.model_text)
        return self._booster

    def predict(self, df: Any) -> Any:
        if self.run_config.task.type != "regression":
            raise VeldraNotImplementedError(
                "Artifact.predict is currently implemented only for task.type='regression'."
            )
        if not isinstance(df, pd.DataFrame):
            raise VeldraValidationError("predict input must be a pandas.DataFrame.")

        feature_names = self.feature_schema.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise VeldraValidationError("feature_schema.feature_names is missing or invalid.")

        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise VeldraValidationError(
                f"Input data is missing required feature columns: {missing}"
            )

        x = df.loc[:, feature_names]
        return np.asarray(self._get_booster().predict(x), dtype=float)

    def simulate(self, df: Any, scenario: dict[str, Any]) -> Any:
        raise VeldraNotImplementedError(
            "Artifact.simulate is not implemented in MVP scaffold."
        )
