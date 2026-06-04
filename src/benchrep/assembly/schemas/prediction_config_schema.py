from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt, model_validator


# -------------------------
# Source config
# -------------------------
class PredictionSourceConfig(BaseModel):
    training_manifest_path: Path
    checkpoint: str = Field(
        default="best",
        description=(
            '"best", "last", or an explicit checkpoint filename from the training '
            'checkpoint directory. Explicit checkpoint filenames must include the '
            '".ckpt" extension, e.g. "epoch=012-step=5473.ckpt".'
        ),
    )


class PredictionDataConfig(BaseModel):
    split: Literal["predict", "train", "val", "test", "all"] = "predict"
    num_workers: NonNegativeInt | None = None
    batch_size: PositiveInt | None = None
    max_batches: PositiveInt | None = None


class PredictionInferenceConfig(BaseModel):
    seed: int | None = None
    seed_workers: bool | None = None
    deterministic: bool | Literal["warn"] | None = None
    float32_matmul_precision: Literal["medium", "high", "highest"] | None = None


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
    seed: int | None = None
    include_input: bool = True
    include_prediction: bool = True


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
    source: PredictionSourceConfig
    data: PredictionDataConfig = Field(default_factory=PredictionDataConfig)
    inference: PredictionInferenceConfig = Field(default_factory=PredictionInferenceConfig)
    exports: PredictionExportConfig = Field(default_factory=PredictionExportConfig)

    @model_validator(mode="after")
    def validate_prediction_config(self) -> PredictionConfig:
        checkpoint = self.source.checkpoint

        if checkpoint not in {"best", "last"} and not checkpoint.endswith(".ckpt"):
            raise ValueError(
                'source.checkpoint must be "best", "last", or a checkpoint '
                'filename ending in ".ckpt".'
            )

        return self