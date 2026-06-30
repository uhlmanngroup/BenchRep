from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Any

from benchrep.interfaces.contracts import (
    ContractKind,
    AutoencoderBatch,
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


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str
    config_model_names: tuple[str, ...]
    model_base_class: SupportedModelBaseClass
    expected_batch_type: type[Any]
    expected_batch_contract_kind: ContractKind
    expected_prediction_output_type: type[Any]
    expected_prediction_output_contract_kind: ContractKind


AUTOENCODER_FAMILY = ModelFamilySpec(
    name="autoencoder",
    config_model_names=("autoencoder", "ae"),
    model_base_class=BenchRepAutoencoderModel,
    expected_batch_type=AutoencoderBatch,
    expected_batch_contract_kind="typeddict",
    expected_prediction_output_type=AutoencoderPredictionOutput,
    expected_prediction_output_contract_kind="dataclass",
)


VAE_FAMILY = ModelFamilySpec(
    name="vae",
    config_model_names=("vae",),
    model_base_class=BenchRepVAEModel,
    expected_batch_type=AutoencoderBatch,
    expected_batch_contract_kind="typeddict",
    expected_prediction_output_type=VAEPredictionOutput,
    expected_prediction_output_contract_kind="dataclass",
)