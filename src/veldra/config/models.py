"""RunConfig models and validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskType = Literal["regression", "binary", "multiclass", "frontier"]
SplitType = Literal["kfold", "stratified", "group", "timeseries"]
TimeSeriesMode = Literal["expanding", "blocked"]
CalibrationType = Literal["platt", "isotonic"]
TuningPreset = Literal["fast", "standard"]
TuningLogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
CausalMethod = Literal["dr", "dr_did"]
CausalDesign = Literal["panel", "repeated_cross_section"]

_TUNE_ALLOWED_OBJECTIVES: dict[str, set[str]] = {
    "regression": {"rmse", "mae", "r2"},
    "binary": {"auc", "logloss", "brier", "accuracy", "f1", "precision", "recall"},
    "multiclass": {"accuracy", "macro_f1", "logloss"},
    "frontier": {"pinball", "pinball_coverage_penalty"},
}
_TUNE_DEFAULT_OBJECTIVES: dict[str, str] = {
    "regression": "rmse",
    "binary": "auc",
    "multiclass": "macro_f1",
    "frontier": "pinball",
}
_CAUSAL_TUNE_ALLOWED_OBJECTIVES: dict[str, set[str]] = {
    "dr": {"dr_std_error", "dr_overlap_penalty", "dr_balance_priority"},
    "dr_did": {
        "drdid_std_error",
        "drdid_overlap_penalty",
        "drdid_balance_priority",
    },
}
_CAUSAL_TUNE_DEFAULT_OBJECTIVES: dict[str, str] = {
    "dr": "dr_balance_priority",
    "dr_did": "drdid_balance_priority",
}


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: TaskType


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | None = None
    target: str
    id_cols: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    drop_cols: list[str] = Field(default_factory=list)


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: SplitType = "kfold"
    n_splits: int = 5
    time_col: str | None = None
    group_col: str | None = None
    seed: int = 42
    timeseries_mode: TimeSeriesMode = "expanding"
    test_size: int | None = None
    gap: int = 0
    embargo: int = 0
    train_size: int | None = None


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lgb_params: dict[str, Any] = Field(default_factory=dict)
    early_stopping_rounds: int | None = 100
    seed: int = 42


class TuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    n_trials: int = 30
    search_space: dict[str, Any] = Field(default_factory=dict)
    preset: TuningPreset = "standard"
    objective: str | None = None
    resume: bool = False
    study_name: str | None = None
    log_level: TuningLogLevel = "INFO"
    coverage_target: float | None = None
    coverage_tolerance: float = 0.01
    penalty_weight: float = 1.0
    causal_penalty_weight: float = 1.0
    causal_balance_threshold: float = 0.10


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calibration: CalibrationType | None = None
    threshold: float | None = None
    threshold_optimization: "ThresholdOptimizationConfig | None" = None


class ThresholdOptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    objective: Literal["f1"] = "f1"


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_dir: str = "artifacts"
    inference_package: bool = False
    onnx_optimization: "OnnxOptimizationConfig" = Field(
        default_factory=lambda: OnnxOptimizationConfig()
    )

    @model_validator(mode="after")
    def _validate_onnx_optimization(self) -> "ExportConfig":
        if self.onnx_optimization.enabled and self.onnx_optimization.mode is None:
            raise ValueError(
                "export.onnx_optimization.mode is required when "
                "export.onnx_optimization.enabled=true"
            )
        return self


class OnnxOptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    mode: Literal["dynamic_quant"] | None = "dynamic_quant"


class FrontierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = 0.90


class CausalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: CausalMethod = "dr"
    treatment_col: str
    estimand: Literal["att", "ate"] = "att"
    design: CausalDesign | None = None
    time_col: str | None = None
    post_col: str | None = None
    unit_id_col: str | None = None
    propensity_clip: float = 0.01
    cross_fit: bool = True
    propensity_calibration: Literal["platt", "isotonic"] = "platt"
    nuisance_params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Single shared entrypoint configuration for all adapters."""

    model_config = ConfigDict(extra="forbid")
    config_version: int
    task: TaskConfig
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    frontier: FrontierConfig = Field(default_factory=FrontierConfig)
    causal: CausalConfig | None = None

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "RunConfig":
        if self.config_version < 1:
            raise ValueError("config_version must be >= 1")

        if self.split.type == "timeseries" and not self.split.time_col:
            raise ValueError("split.time_col is required when split.type='timeseries'")

        if self.split.type == "group" and not self.split.group_col:
            raise ValueError("split.group_col is required when split.type='group'")

        if self.split.type == "timeseries":
            if self.split.gap < 0:
                raise ValueError("split.gap must be >= 0 when split.type='timeseries'")
            if self.split.embargo < 0:
                raise ValueError("split.embargo must be >= 0 when split.type='timeseries'")
            if self.split.test_size is not None and self.split.test_size < 1:
                raise ValueError("split.test_size must be >= 1 when split.type='timeseries'")
            if self.split.timeseries_mode == "blocked":
                if self.split.train_size is None or self.split.train_size < 1:
                    raise ValueError(
                        "split.train_size must be >= 1 when "
                        "split.type='timeseries' and split.timeseries_mode='blocked'"
                    )
            else:
                if self.split.train_size is not None:
                    raise ValueError(
                        "split.train_size can be set only when "
                        "split.type='timeseries' and split.timeseries_mode='blocked'"
                    )
        else:
            if self.split.timeseries_mode != "expanding":
                raise ValueError(
                    "split.timeseries_mode can be customized only when split.type='timeseries'"
                )
            if self.split.test_size is not None:
                raise ValueError(
                    "split.test_size can be customized only when split.type='timeseries'"
                )
            if self.split.gap != 0:
                raise ValueError("split.gap can be customized only when split.type='timeseries'")
            if self.split.embargo != 0:
                raise ValueError(
                    "split.embargo can be customized only when split.type='timeseries'"
                )
            if self.split.train_size is not None:
                raise ValueError(
                    "split.train_size can be customized only when split.type='timeseries'"
                )

        if self.task.type == "frontier":
            if not (0.0 < self.frontier.alpha < 1.0):
                raise ValueError("frontier.alpha must satisfy 0 < alpha < 1 for frontier task")
            if self.split.type == "stratified":
                raise ValueError("split.type='stratified' is not supported for frontier task")
        else:
            if self.frontier.alpha != 0.90:
                raise ValueError(
                    "frontier settings can only be customized when task.type='frontier'"
                )

        if self.task.type != "binary":
            if (
                self.postprocess.calibration is not None
                or self.postprocess.threshold is not None
                or self.postprocess.threshold_optimization is not None
            ):
                raise ValueError(
                    "postprocess.calibration/threshold/threshold_optimization can only be set "
                    "when task.type='binary'"
                )
        else:
            if self.postprocess.calibration is not None and self.postprocess.calibration != "platt":
                raise ValueError("binary calibration supports only 'platt' in current phase")
            if self.postprocess.threshold is not None and not (
                0.0 <= self.postprocess.threshold <= 1.0
            ):
                raise ValueError("postprocess.threshold must be between 0 and 1")
            if (
                self.postprocess.threshold_optimization is not None
                and self.postprocess.threshold_optimization.enabled
                and self.postprocess.threshold is not None
            ):
                raise ValueError(
                    "postprocess.threshold cannot be combined with enabled "
                    "postprocess.threshold_optimization"
                )

        if self.causal is None:
            if self.task.type in _TUNE_ALLOWED_OBJECTIVES:
                objective = self.tuning.objective
                if (
                    objective is not None
                    and objective not in _TUNE_ALLOWED_OBJECTIVES[self.task.type]
                ):
                    allowed = sorted(_TUNE_ALLOWED_OBJECTIVES[self.task.type])
                    raise ValueError(
                        f"tuning.objective '{objective}' is not allowed for task.type="
                        f"'{self.task.type}'. Allowed: {allowed}"
                    )
            if self.tuning.causal_penalty_weight != 1.0:
                raise ValueError(
                    "tuning.causal_penalty_weight can only be customized when causal is configured"
                )
            if self.tuning.causal_balance_threshold != 0.10:
                raise ValueError(
                    "tuning.causal_balance_threshold can only be customized when causal is "
                    "configured"
                )
        else:
            allowed = _CAUSAL_TUNE_ALLOWED_OBJECTIVES[self.causal.method]
            objective = self.tuning.objective
            if objective is not None and objective not in allowed:
                raise ValueError(
                    f"tuning.objective '{objective}' is not allowed for causal.method="
                    f"'{self.causal.method}'. Allowed: {sorted(allowed)}"
                )
            if self.tuning.causal_penalty_weight < 0:
                raise ValueError("tuning.causal_penalty_weight must be >= 0")
            if self.tuning.causal_balance_threshold <= 0:
                raise ValueError("tuning.causal_balance_threshold must be > 0")

        if self.task.type == "frontier":
            resolved_target = (
                self.frontier.alpha
                if self.tuning.coverage_target is None
                else self.tuning.coverage_target
            )
            if not (0.0 < resolved_target < 1.0):
                raise ValueError(
                    "tuning.coverage_target must satisfy 0 < value < 1 for frontier task"
                )
            if self.tuning.coverage_tolerance < 0:
                raise ValueError("tuning.coverage_tolerance must be >= 0 for frontier task")
            if self.tuning.penalty_weight < 0:
                raise ValueError("tuning.penalty_weight must be >= 0 for frontier task")
        else:
            if self.tuning.coverage_target is not None:
                raise ValueError("tuning.coverage_target can only be set when task.type='frontier'")
            if self.tuning.coverage_tolerance != 0.01:
                raise ValueError(
                    "tuning.coverage_tolerance can only be customized when task.type='frontier'"
                )
            if self.tuning.penalty_weight != 1.0:
                raise ValueError(
                    "tuning.penalty_weight can only be customized when task.type='frontier'"
                )

        if self.causal is not None:
            if not (0.0 < self.causal.propensity_clip < 0.5):
                raise ValueError("causal.propensity_clip must satisfy 0 < value < 0.5")
            if self.causal.treatment_col == self.data.target:
                raise ValueError("causal.treatment_col must differ from data.target")
            if self.causal.method == "dr":
                if self.task.type not in {"regression", "binary"}:
                    raise ValueError(
                        "causal.method='dr' supports only task.type='regression' or 'binary'"
                    )
                if self.causal.design is not None:
                    raise ValueError("causal.design is only supported for causal.method='dr_did'")
                if self.causal.time_col is not None:
                    raise ValueError("causal.time_col is only supported for causal.method='dr_did'")
                if self.causal.post_col is not None:
                    raise ValueError("causal.post_col is only supported for causal.method='dr_did'")
                if self.causal.unit_id_col is not None:
                    raise ValueError(
                        "causal.unit_id_col is only supported for causal.method='dr_did'"
                    )
            elif self.causal.method == "dr_did":
                if self.task.type not in {"regression", "binary"}:
                    raise ValueError(
                        "causal.method='dr_did' supports only "
                        "task.type='regression' or 'binary' in current phase"
                    )
                if self.task.type == "binary" and self.causal.estimand != "att":
                    raise ValueError(
                        "causal.estimand must be 'att' when "
                        "causal.method='dr_did' and task.type='binary'"
                    )
                if self.causal.design is None:
                    raise ValueError("causal.design is required when causal.method='dr_did'")
                if self.causal.time_col is None:
                    raise ValueError("causal.time_col is required when causal.method='dr_did'")
                if self.causal.post_col is None:
                    raise ValueError("causal.post_col is required when causal.method='dr_did'")
                if self.causal.design == "panel" and self.causal.unit_id_col is None:
                    raise ValueError(
                        "causal.unit_id_col is required for causal.method='dr_did' "
                        "and causal.design='panel'"
                    )

        return self


def resolve_tuning_objective(
    task_type: str,
    objective: str | None,
    *,
    causal_method: str | None = None,
) -> str:
    if causal_method is not None:
        if causal_method not in _CAUSAL_TUNE_ALLOWED_OBJECTIVES:
            raise ValueError(f"Unsupported causal method for tuning objective: '{causal_method}'")
        if objective is None:
            return _CAUSAL_TUNE_DEFAULT_OBJECTIVES[causal_method]
        if objective not in _CAUSAL_TUNE_ALLOWED_OBJECTIVES[causal_method]:
            allowed = sorted(_CAUSAL_TUNE_ALLOWED_OBJECTIVES[causal_method])
            raise ValueError(
                f"tuning.objective '{objective}' is not allowed for causal.method="
                f"'{causal_method}'. Allowed: {allowed}"
            )
        return objective

    if task_type not in _TUNE_ALLOWED_OBJECTIVES:
        raise ValueError(f"Unsupported task type for tuning objective: '{task_type}'")
    if objective is None:
        return _TUNE_DEFAULT_OBJECTIVES[task_type]
    if objective not in _TUNE_ALLOWED_OBJECTIVES[task_type]:
        allowed = sorted(_TUNE_ALLOWED_OBJECTIVES[task_type])
        raise ValueError(
            f"tuning.objective '{objective}' is not allowed for task.type='{task_type}'. "
            f"Allowed: {allowed}"
        )
    return objective
