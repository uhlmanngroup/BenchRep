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
class LossConfig(NamedConfig):
    weight: float = 1.0


class OptimizerConfig(NamedConfig):
    pass


# -------------------------
# Training/runtime configuration
# -------------------------
class TrainerConfig(BaseModel):
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
    model: ModelConfig
    encoder: EncoderConfig
    decoder: DecoderConfig | None = None
    losses: dict[str, LossConfig] = Field(default_factory=dict)
    optimizer: OptimizerConfig
    data: DataConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    seed: int = 137

    @model_validator(mode="after")
    def validate_model_requirements(self) -> "BenchRepConfig":
        model_name = self.model.name.strip().lower().replace("-", "_")

        if model_name == "autoencoder":
            if self.decoder is None:
                raise ValueError("Autoencoder requires a decoder config section.")

            if "reconstruction" not in self.losses:
                raise ValueError(
                    "Autoencoder requires a reconstruction loss under "
                    "`losses.reconstruction`."
                )

        return self