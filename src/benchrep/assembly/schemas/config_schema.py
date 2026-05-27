from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# -------------------------
# Generic reusable blocks
# -------------------------
class NamedConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Run/output configuration
# -------------------------
class RunConfig(BaseModel):
    output_root: Path = Path("outputs")
    project_name: str | None = None


# -------------------------
# Architecture configuration
# -------------------------
class ModelConfig(NamedConfig):
    pass


class EncoderConfig(NamedConfig):
    pass


class DecoderConfig(NamedConfig):
    pass


# -------------------------
# Optimization/loss configuration
# -------------------------
class LossesConfig(BaseModel):
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(NamedConfig):
    pass


# -------------------------
# Training/runtime configuration
# -------------------------
class ReproducibilityConfig(BaseModel):
    seed: int | None = 137
    seed_workers: bool = True
    float32_matmul_precision: Literal["medium", "high", "highest"] | None = "highest"


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class LoggerConfig(NamedConfig):
    credential_path: Path | None = None
    model_config = ConfigDict(extra="allow")


# -------------------------
# Data configuration
# -------------------------
class TransformConfig(NamedConfig):
    pass


class DatasetConfig(BaseModel):
    name: str
    root: Path | None = None
    download: bool | None = None
    transform: TransformConfig | None = None


class DataModuleConfig(BaseModel):
    batch_size: int = 32
    val_fraction: float = 0.1
    num_workers: int = 4
    pin_memory: bool | Literal["auto"] = "auto"
    persistent_workers: bool = False
    drop_last: bool = False


class DataConfig(BaseModel):
    dataset: DatasetConfig
    datamodule: DataModuleConfig


# -------------------------
# Full experiment configuration
# -------------------------
class BenchRepConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    model: ModelConfig
    encoder: EncoderConfig
    decoder: DecoderConfig | None = None
    losses: dict[str, dict[str, LossesConfig]] = Field(default_factory=dict)
    optimizer: OptimizerConfig
    dataset: DatasetConfig
    datamodule: DataModuleConfig = Field(default_factory=DataModuleConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    logger: LoggerConfig | None = None

    @model_validator(mode="after")
    def validate_model_requirements(self) -> "BenchRepConfig":
        model_name = self.model.name.strip().lower().replace("-", "_")

        if model_name == "autoencoder":
            if self.decoder is None:
                raise ValueError("Autoencoder requires a decoder config section.")

            if "reconstruction" not in self.losses or not self.losses["reconstruction"]:
                raise ValueError(
                    "Autoencoder requires at least one reconstruction loss under "
                    "`losses.reconstruction`."
                )

        return self