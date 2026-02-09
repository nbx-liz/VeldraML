"""RunConfig models and validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskType = Literal["regression", "binary", "multiclass", "frontier"]
SplitType = Literal["kfold", "stratified", "group", "timeseries"]
CalibrationType = Literal["platt", "isotonic"]
TuningPreset = Literal["fast", "standard"]


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

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> "RunConfig":
        if self.config_version < 1:
            raise ValueError("config_version must be >= 1")

        if self.split.type == "timeseries" and not self.split.time_col:
            raise ValueError("split.time_col is required when split.type='timeseries'")

        if self.split.type == "group" and not self.split.group_col:
            raise ValueError("split.group_col is required when split.type='group'")

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
            if (
                self.postprocess.calibration is not None
                and self.postprocess.calibration != "platt"
            ):
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

        return self
