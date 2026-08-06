from __future__ import annotations

from pathlib import Path
from typing import Literal, Annotated

from pydantic import (
    ConfigDict,
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
    model_validator,
    field_validator,
    ValidationInfo,
    StringConstraints,
)

from benchrep.assembly.schemas.training_config_schema import (
    NamedConfig,
    SupportedDatasetConfig,
)


# -------------------------
# Generic reusable blocks
# -------------------------
class _PredictionConfigBaseModel(BaseModel):
    """Base model for strict prediction configuration schemas."""

    model_config = ConfigDict(extra="forbid")


# -------------------------
# Source config
# -------------------------
class PredictionSourceConfig(_PredictionConfigBaseModel):
    training_manifest_path: Path | None = None
    checkpoint: Literal["best", "last"] | Path = Field(
        default="best",
        description=(
            '"best", "last", a checkpoint filename (e.g. "epoch=042-step=5698.ckpt") '
            'from the training checkpoint directory, or an absolute checkpoint path.'
        ),
    )

    @field_validator("checkpoint")
    @classmethod
    def validate_checkpoint(
        cls,
        value: Literal["best", "last"] | Path,
    ) -> Literal["best", "last"] | Path:
        if value in {"best", "last"}:
            return value

        assert isinstance(value, Path)

        checkpoint_path = value.expanduser()

        if (
            not checkpoint_path.is_absolute()
            and checkpoint_path.parent != Path(".")
        ):
            raise ValueError(
                "checkpoint must be 'best', 'last', a checkpoint filename, "
                "or an absolute checkpoint path; directory-containing relative "
                "paths are not supported."
            )

        if checkpoint_path.suffix != ".ckpt":
            raise ValueError(
                "checkpoint filenames and paths must end with '.ckpt'."
            )

        if checkpoint_path.is_absolute():
            return checkpoint_path.resolve()

        return checkpoint_path


class PredictionDataConfig(_PredictionConfigBaseModel):
    num_workers: NonNegativeInt | None = None
    batch_size: PositiveInt | None = None
    max_batches: PositiveInt | None = None


class PredictionInferenceConfig(_PredictionConfigBaseModel):
    seed: int | None = None
    seed_workers: bool | None = None
    deterministic: bool | Literal["warn"] | None = None
    float32_matmul_precision: Literal["medium", "high", "highest"] | None = None
    reconstruction_latent_source: Literal["mean", "sample"] | None = None


# -------------------------
# Transforms config
# -------------------------
class PredictionTransformConfig(NamedConfig):
    """Configuration for one transform in an ordered prediction sequence.

    Every transform declared here applies during prediction, so split-targeting
    metadata is unnecessary. Use `benchrep.inspect_registry("transform")` to
    inspect available names and aliases, and
    `benchrep.inspect_registry("transform", "<name>")` for the registered
    constructor signature and documentation.
    """


# -------------------------
# Exports config
# -------------------------
class PredictionEmbeddingsExportConfig(_PredictionConfigBaseModel):
    enabled: bool = True
    keys: list[str] | Literal["auto"] = "auto"
    primary_key: str | Literal["auto"] = "auto"


class PredictionReconstructionsExportConfig(_PredictionConfigBaseModel):
    enabled: bool = True
    n_examples: Literal["all"] | PositiveInt = 32
    selection: Literal["first", "random"] = "first"
    stratify_by: Annotated[
                          str,
                          StringConstraints(strip_whitespace=True, min_length=1),
                      ] | None = None
    seed: int | None = None
    include_input: bool = True
    include_prediction: bool = True

    @model_validator(mode="after")
    def validate_sampling(self) -> PredictionReconstructionsExportConfig:
        if self.stratify_by is not None and self.selection != "random":
            raise ValueError(
                "`exports.reconstructions.selection` must be 'random' when "
                "`exports.reconstructions.stratify_by` is set."
            )

        return self


class PredictionExportConfig(_PredictionConfigBaseModel):
    mode: Literal["standard", "all", "custom"] = "standard"
    embeddings: PredictionEmbeddingsExportConfig = Field(
        default_factory=PredictionEmbeddingsExportConfig)
    reconstructions: PredictionReconstructionsExportConfig = Field(
        default_factory=PredictionReconstructionsExportConfig
    )


# -------------------------
# Full prediction configuration
# -------------------------
class PredictionConfig(_PredictionConfigBaseModel):
    stage: Literal["prediction"] = "prediction"
    source: PredictionSourceConfig = Field(
        default_factory=PredictionSourceConfig)
    dataset: SupportedDatasetConfig | None = None
    data: PredictionDataConfig = Field(default_factory=PredictionDataConfig)
    inference: PredictionInferenceConfig = Field(default_factory=PredictionInferenceConfig)
    transforms: list[PredictionTransformConfig] | None = Field(
        default=None,
        description=(
            "Ordered transforms applied during prediction. If omitted or null, "
            "validation-targeted transforms are inherited from the resolved "
            "training config when available. If training used an external "
            "datamodule, no configured transforms are inherited. An explicit "
            "list replaces the inherited transforms, and an empty list applies "
            "no configured transforms."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Inherits available validation-targeted training transforms; "
                "otherwise uses no configured transforms."
            ),
            "null_behavior": (
                "Inherits available validation-targeted training transforms; "
                "otherwise uses no configured transforms."
            ),
        },
    )
    exports: PredictionExportConfig = Field(default_factory=PredictionExportConfig)

    @model_validator(mode="after")
    def validate_prediction_config(self, info: ValidationInfo) -> "PredictionConfig":
        ctx = info.context or {}
        training_manifest_path_overridden = ctx.get(
            "training_manifest_path_overridden",
            False,
        )

        if self.source.training_manifest_path is None:
            if not training_manifest_path_overridden:
                raise ValueError(
                    "`source.training_manifest_path` is required unless "
                    "`training_manifest_path` is passed to predict()."
                )
        elif self.source.training_manifest_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("source.training_manifest_path must point to a YAML file.")

        return self