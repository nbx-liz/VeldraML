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
from veldra.simulate import apply_scenario, build_simulation_frame


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
        threshold_curve: pd.DataFrame | None = None,
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
        self.threshold_curve = threshold_curve
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
        threshold_curve: pd.DataFrame | None = None,
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
            threshold_curve=threshold_curve,
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
            threshold_curve=extras.get("threshold_curve"),
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
            threshold_curve=self.threshold_curve,
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

    def _predict_multiclass(self, x: pd.DataFrame) -> pd.DataFrame:
        target_classes = self.feature_schema.get("target_classes")
        if not isinstance(target_classes, list) or len(target_classes) < 3:
            raise VeldraArtifactError(
                "Multiclass artifact requires feature_schema.target_classes with at least 3 labels."
            )

        raw = np.asarray(self._get_booster().predict(x), dtype=float)
        num_class = len(target_classes)
        if raw.ndim == 1:
            if raw.size != len(x) * num_class:
                raise VeldraArtifactError(
                    "Multiclass prediction output has invalid shape for configured classes."
                )
            raw = raw.reshape(len(x), num_class)
        if raw.ndim != 2 or raw.shape[1] != num_class:
            raise VeldraArtifactError(
                "Multiclass prediction output has invalid dimensions."
            )

        proba = np.clip(raw, 1e-7, 1 - 1e-7)
        row_sum = proba.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise VeldraArtifactError("Multiclass prediction probabilities have invalid row sums.")
        proba = proba / row_sum

        label_idx = np.argmax(proba, axis=1)
        label_pred = [target_classes[int(idx)] for idx in label_idx]

        payload: dict[str, Any] = {"label_pred": label_pred}
        for idx, class_label in enumerate(target_classes):
            payload[f"proba_{class_label}"] = proba[:, idx]
        return pd.DataFrame(payload, index=x.index)

    def _predict_frontier(self, source_df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        pred = np.asarray(self._get_booster().predict(x), dtype=float)
        payload: dict[str, Any] = {"frontier_pred": pred}

        target_col = self.feature_schema.get("target", self.run_config.data.target)
        if isinstance(target_col, str) and target_col in source_df.columns:
            try:
                y_obs = source_df[target_col].to_numpy(dtype=float)
            except Exception as exc:
                raise VeldraValidationError(
                    f"Frontier target column '{target_col}' must be numeric for u_hat output."
                ) from exc
            payload["u_hat"] = np.maximum(0.0, pred - y_obs)

        return pd.DataFrame(payload, index=x.index)

    def predict(self, df: Any) -> Any:
        if not isinstance(df, pd.DataFrame):
            raise VeldraValidationError("predict input must be a pandas.DataFrame.")
        x = self._prepare_feature_frame(df)

        if self.run_config.task.type == "regression":
            return self._predict_regression(x)
        if self.run_config.task.type == "binary":
            return self._predict_binary(x)
        if self.run_config.task.type == "multiclass":
            return self._predict_multiclass(x)
        if self.run_config.task.type == "frontier":
            return self._predict_frontier(df, x)
        raise VeldraNotImplementedError(
            "Artifact.predict is currently implemented only for regression, binary, multiclass, "
            "and frontier tasks."
        )

    def simulate(self, df: Any, scenario: dict[str, Any]) -> Any:
        if self.run_config.task.type not in {"regression", "binary", "multiclass", "frontier"}:
            raise VeldraNotImplementedError(
                "Artifact.simulate is currently implemented only for regression, binary, "
                "multiclass, and frontier tasks."
            )
        if not isinstance(df, pd.DataFrame):
            raise VeldraValidationError("simulate input must be a pandas.DataFrame.")
        if df.empty:
            raise VeldraValidationError("simulate input DataFrame is empty.")
        if not isinstance(scenario, dict):
            raise VeldraValidationError("scenario must be a dict.")

        scenario_name = scenario.get("name")
        if not isinstance(scenario_name, str) or not scenario_name.strip():
            scenario_name = "scenario_1"
        scenario_payload = {
            "name": scenario_name,
            "actions": scenario.get("actions"),
        }
        modified = apply_scenario(
            df,
            scenario_payload,
            target_col=self.run_config.data.target,
            id_cols=self.run_config.data.id_cols,
        )

        base_pred = self.predict(df)
        scenario_pred = self.predict(modified)
        return build_simulation_frame(
            task_type=self.run_config.task.type,
            row_ids=df.index,
            scenario_name=scenario_name,
            base_pred=base_pred,
            scenario_pred=scenario_pred,
            target_classes=self.feature_schema.get("target_classes"),
        )
