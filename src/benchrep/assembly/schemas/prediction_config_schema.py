from __future__ import annotations

from pathlib import Path
from typing import Literal, Annotated

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeInt,
    model_validator,
    ValidationInfo,
    StringConstraints,
)

from benchrep.assembly.schemas.training_config_schema import (
    TransformConfig,
    SupportedDatasetConfig,
)


# -------------------------
# Source config
# -------------------------
class PredictionSourceConfig(BaseModel):
    training_manifest_path: Path | None = None
    checkpoint: str = Field(
        default="best",
        description=(
            '"best", "last", or an explicit checkpoint filename from the training '
            'checkpoint directory. Explicit checkpoint filenames must include the '
            '".ckpt" extension, e.g. "epoch=012-step=5473.ckpt".'
        ),
    )


class PredictionDataConfig(BaseModel):
    num_workers: NonNegativeInt | None = None
    batch_size: PositiveInt | None = None
    max_batches: PositiveInt | None = None


class PredictionInferenceConfig(BaseModel):
    seed: int | None = None
    seed_workers: bool | None = None
    deterministic: bool | Literal["warn"] | None = None
    float32_matmul_precision: Literal["medium", "high", "highest"] | None = None


# -------------------------
# Transforms config
# -------------------------
class PredictionTransformConfig(TransformConfig):
    """Configuration for one transform in an ordered prediction sequence."""

    category: Literal["preprocessing"] = Field(
        default="preprocessing",
        description=(
            "Prediction transform category. Defaults to preprocessing because "
            "all configured prediction transforms are applied to prediction samples."
        ),
    )


# -------------------------
# Exports config
# -------------------------
class PredictionEmbeddingsExportConfig(BaseModel):
    enabled: bool = True
    keys: list[str] | Literal["auto"] = "auto"
    primary_key: str | Literal["auto"] = "auto"


class PredictionReconstructionsExportConfig(BaseModel):
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


class PredictionExportConfig(BaseModel):
    mode: Literal["standard", "all", "custom"] = "standard"
    embeddings: PredictionEmbeddingsExportConfig = Field(
        default_factory=PredictionEmbeddingsExportConfig)
    reconstructions: PredictionReconstructionsExportConfig = Field(
        default_factory=PredictionReconstructionsExportConfig
    )


# -------------------------
# Full prediction configuration
# -------------------------
class PredictionConfig(BaseModel):
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
            "preprocessing transforms are inherited from the resolved training "
            "config when available; if training used an external datamodule, "
            "no configured transforms are applied. An explicit list replaces "
            "the inherited transforms, and an empty list applies no configured "
            "transforms."
        ),
        json_schema_extra={
            "omit_behavior": (
                "Inherits available training preprocessing transforms; otherwise "
                "uses no configured transforms."
            ),
            "null_behavior": (
                "Inherits available training preprocessing transforms; otherwise "
                "uses no configured transforms."
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

        checkpoint = self.source.checkpoint

        if self.source.training_manifest_path is None:
            if not training_manifest_path_overridden:
                raise ValueError(
                    "`source.training_manifest_path` is required unless "
                    "`training_manifest_path` is passed to predict()."
                )
        elif self.source.training_manifest_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("source.training_manifest_path must point to a YAML file.")

        if checkpoint not in {"best", "last"} and not checkpoint.endswith(".ckpt"):
            raise ValueError(
                'source.checkpoint must be "best", "last", or a checkpoint '
                'filename ending in ".ckpt".'
            )

        return self