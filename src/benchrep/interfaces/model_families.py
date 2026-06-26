from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchrep.architecture.models import (
    AutoencoderPredictionOutput,
    VAEPredictionOutput,
)


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    config_model_names: tuple[str, ...]
    prediction_output_type: type[Any]


AUTOENCODER_FAMILY = ModelFamilySpec(
    name="autoencoder",
    config_model_names=("autoencoder", "ae"),
    prediction_output_type=AutoencoderPredictionOutput,
)


VAE_FAMILY = ModelFamilySpec(
    name="vae",
    config_model_names=("vae",),
    prediction_output_type=VAEPredictionOutput,
)