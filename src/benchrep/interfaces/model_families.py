from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from benchrep.interfaces.contracts import (
    AutoencoderPredictionOutput,
    VAEPredictionOutput,
)
from benchrep.interfaces.models import (
    BenchRepAutoencoderModel,
    BenchRepVAEModel,
)


SupportedModel: TypeAlias = BenchRepAutoencoderModel | BenchRepVAEModel

SupportedModelBaseClass: TypeAlias = (
    type[BenchRepAutoencoderModel] | type[BenchRepVAEModel]
)

SupportedPredictionOutputType: TypeAlias = (
    type[AutoencoderPredictionOutput] | type[VAEPredictionOutput]
)


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    config_model_names: tuple[str, ...]
    model_base_class: SupportedModelBaseClass
    prediction_output_type: SupportedPredictionOutputType


AUTOENCODER_FAMILY = ModelFamilySpec(
    name="autoencoder",
    config_model_names=("autoencoder", "ae"),
    model_base_class=BenchRepAutoencoderModel,
    prediction_output_type=AutoencoderPredictionOutput,
)


VAE_FAMILY = ModelFamilySpec(
    name="vae",
    config_model_names=("vae",),
    model_base_class=BenchRepVAEModel,
    prediction_output_type=VAEPredictionOutput,
)