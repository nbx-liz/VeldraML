"""Stable Artifact API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from veldra import __version__
from veldra.api.exceptions import (
    VeldraArtifactError,
    VeldraNotImplementedError,
    VeldraValidationError,
)
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
        calibrator: Any | None = None,
        calibration_curve: pd.DataFrame | None = None,
        threshold: dict[str, Any] | None = None,
    ) -> None:
        self.run_config = run_config
        self.manifest = manifest
        self.feature_schema = feature_schema or {}
        self.model_text = model_text
        self.metrics = metrics or {}
        self.cv_results = cv_results
        self.calibrator = calibrator
        self.calibration_curve = calibration_curve
        self.threshold = threshold or {"policy": "fixed", "value": 0.5}
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
        calibrator: Any | None = None,
        calibration_curve: pd.DataFrame | None = None,
        threshold: dict[str, Any] | None = None,
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
            calibrator=calibrator,
            calibration_curve=calibration_curve,
            threshold=threshold,
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
            calibrator=extras.get("calibrator"),
            calibration_curve=extras.get("calibration_curve"),
            threshold=extras.get("threshold"),
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
            calibrator=self.calibrator,
            calibration_curve=self.calibration_curve,
            threshold=self.threshold,
        )

    def _get_booster(self) -> lgb.Booster:
        if self.model_text is None:
            raise VeldraValidationError("Artifact model is missing. Run fit before predict.")
        if self._booster is None:
            self._booster = lgb.Booster(model_str=self.model_text)
        return self._booster

    def _prepare_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_names = self.feature_schema.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise VeldraValidationError("feature_schema.feature_names is missing or invalid.")

        missing = [name for name in feature_names if name not in df.columns]
        if missing:
            raise VeldraValidationError(
                f"Input data is missing required feature columns: {missing}"
            )
        return df.loc[:, feature_names]

    def _predict_regression(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._get_booster().predict(x), dtype=float)

    def _predict_binary(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.calibrator is None:
            raise VeldraArtifactError("Binary artifact is missing calibrator.pkl.")
        raw = np.asarray(self._get_booster().predict(x), dtype=float)
        raw = np.clip(raw, 1e-7, 1 - 1e-7)
        cal = np.asarray(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
        threshold_value = float((self.threshold or {}).get("value", 0.5))
        label_pred = (cal >= threshold_value).astype(int)
        return pd.DataFrame(
            {
                "p_cal": cal,
                "p_raw": raw,
                "label_pred": label_pred,
            },
            index=x.index,
        )

    def predict(self, df: Any) -> Any:
        if not isinstance(df, pd.DataFrame):
            raise VeldraValidationError("predict input must be a pandas.DataFrame.")
        x = self._prepare_feature_frame(df)

        if self.run_config.task.type == "regression":
            return self._predict_regression(x)
        if self.run_config.task.type == "binary":
            return self._predict_binary(x)
        raise VeldraNotImplementedError(
            "Artifact.predict is currently implemented only for regression and binary tasks."
        )

    def simulate(self, df: Any, scenario: dict[str, Any]) -> Any:
        raise VeldraNotImplementedError(
            "Artifact.simulate is not implemented in MVP scaffold."
        )
