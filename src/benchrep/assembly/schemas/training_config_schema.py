from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    NonNegativeInt,
    model_validator,
    ValidationInfo,
    Discriminator,
    Tag,
    field_validator,
)

from benchrep.assembly.registries.core import MODELS
from benchrep.assembly.registries.utils import normalize_name
from benchrep.architecture.models import (
    Autoencoder,
    VAE,
)


# Helper for model and datamodule overrides
def _require_present(value: object, field_name: str) -> None:
    if value is None:
        raise ValueError(
            f"`{field_name}` is required unless the corresponding object is overridden."
        )


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
class LossTermConfig(BaseModel):
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
    max_epochs: PositiveInt | None = None
    accelerator: str | None = "auto"
    devices: str | int | list[int] | None = "auto"
    log_every_n_steps: PositiveInt | None = None
    deterministic: bool | Literal["warn"] | None = None
    benchmark: bool | None = None
    precision: str | int | None = None

    model_config = ConfigDict(extra="allow")


class LoggerConfig(NamedConfig):
    credential_path: Path | None = None
    model_config = ConfigDict(extra="allow")


class CheckpointConfig(BaseModel):
    monitor: str | None = "val/loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 1
    save_last: bool = True
    filename: str = "{epoch:03d}-{step}"


# -------------------------
# Inspection configuration
# -------------------------
class TorchviewConfig(BaseModel):
    enabled: bool = False
    expand_nested: bool = True
    depth: int = Field(default=10, ge=0)


class InspectionConfig(BaseModel):
    torchview: TorchviewConfig = Field(default_factory=TorchviewConfig)


# -------------------------
# Data configuration
# -------------------------
ParamsT = TypeVar("ParamsT")


class TransformConfig(NamedConfig):
    pass


class DatasetConfig(BaseModel, Generic[ParamsT]):
    name: str
    params: ParamsT

    @field_validator("name", mode="before")
    @classmethod
    def normalize_dataset_name(cls, value: Any) -> str:
        return normalize_name(value, field_name="dataset.name")


class MNISTDatasetParams(BaseModel):
    root: Path
    split: Literal["train", "test"] = "train"
    download: bool = False
    transform: TransformConfig | None = None


class MNISTDatasetConfig(DatasetConfig[MNISTDatasetParams]):
    name: Literal["mnist"] = "mnist"
    params: MNISTDatasetParams


class CustomDatasetConfig(DatasetConfig[dict[str, Any]]):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


def _dataset_config_discriminator(value: Any) -> str:
    if isinstance(value, dict):
        name = value.get("name")
    else:
        name = getattr(value, "name", None)

    if isinstance(name, str) and name.strip():
        normalized_name = normalize_name(name, field_name="dataset.name")

        if normalized_name == "mnist":
            return "mnist"

    return "custom"


SupportedDatasetConfig = Annotated[
    Annotated[MNISTDatasetConfig, Tag("mnist")]
    | Annotated[CustomDatasetConfig, Tag("custom")],
    Discriminator(_dataset_config_discriminator),
]


class DataModuleConfig(BaseModel):
    batch_size: PositiveInt = 32
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    num_workers: NonNegativeInt = 4
    pin_memory: bool | Literal["auto"] = "auto"
    persistent_workers: bool = False
    drop_last: bool = False


# -------------------------
# Full experiment configuration
# -------------------------
class TrainingConfig(BaseModel):
    stage: Literal["training"] = "training"
    run: RunConfig = Field(default_factory=RunConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)

    model: ModelConfig | None = None
    encoder: EncoderConfig | None = None
    decoder: DecoderConfig | None = None
    losses: dict[str, dict[str, LossTermConfig]] | None = Field(default_factory=dict)
    optimizer: OptimizerConfig | None = None

    dataset: SupportedDatasetConfig | None = None
    datamodule: DataModuleConfig | None = Field(default_factory=DataModuleConfig)

    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    logger: LoggerConfig | None = None
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    inspection: InspectionConfig = Field(default_factory=InspectionConfig)

    @model_validator(mode="after")
    def validate_override_requirements(self, info: ValidationInfo) -> "TrainingConfig":
        ctx = info.context or {}
        model_overridden = ctx.get("model_overridden", False)
        datamodule_overridden = ctx.get("datamodule_overridden", False)

        if not model_overridden:
            _require_present(self.model, "model")
            _require_present(self.encoder, "encoder")
            _require_present(self.losses, "losses")
            _require_present(self.optimizer, "optimizer")

        if not datamodule_overridden:
            _require_present(self.dataset, "dataset")
            _require_present(self.datamodule, "datamodule")

        return self

    @model_validator(mode="after")
    def validate_model_requirements(self, info: ValidationInfo) -> "TrainingConfig":
        ctx = info.context or {}
        model_overridden = ctx.get("model_overridden", False)

        if model_overridden:
            return self

        assert self.model is not None
        assert self.encoder is not None
        assert self.losses is not None
        assert self.optimizer is not None

        model_name = normalize_name(
            self.model.name,
            field_name="model.name",
        )

        model_cls = MODELS.get(model_name)

        if model_cls is Autoencoder:
            if self.decoder is None:
                raise ValueError("Autoencoder requires a decoder config section.")

            if "reconstruction" not in self.losses or not self.losses["reconstruction"]:
                raise ValueError(
                    "Autoencoder requires at least one reconstruction loss under "
                    "`losses.reconstruction`."
                )

        elif model_cls is VAE:
            if self.decoder is None:
                raise ValueError("VAE requires a decoder config section.")

            if "latent_dim" not in self.model.params:
                raise ValueError("VAE requires `model.params.latent_dim`.")

            latent_dim = self.model.params.get("latent_dim")
            if not isinstance(latent_dim, int) or latent_dim <= 0:
                raise ValueError(
                    "VAE requires `model.params.latent_dim` to be a positive integer."
                )

            if "reconstruction" not in self.losses or not self.losses["reconstruction"]:
                raise ValueError(
                    "VAE requires at least one reconstruction loss under "
                    "`losses.reconstruction`."
                )

            if "regularization" not in self.losses or not self.losses["regularization"]:
                raise ValueError(
                    "VAE requires at least one regularization loss under "
                    "`losses.regularization`."
                )

        return self

    @model_validator(mode="after")
    def validate_checkpointing(self) -> "TrainingConfig":
        if self.checkpointing.monitor is None and not self.checkpointing.save_last:
            raise ValueError(
                "checkpointing.monitor=None requires checkpointing.save_last=True, "
                "otherwise no checkpoint would be saved."
            )

        return self