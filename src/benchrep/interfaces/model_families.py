from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import TypeAlias, Any, cast

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
from benchrep.architecture.models.composite import CompositeModel


SupportedModel: TypeAlias = (
        BenchRepAutoencoderModel
        | BenchRepVAEModel
        | CompositeModel
)

SupportedModelBaseClass: TypeAlias = (
    type[BenchRepAutoencoderModel] | type[BenchRepVAEModel]
)


@dataclass(frozen=True)
class ModelFamilySpec:
    name: str


@dataclass(frozen=True)
class CanonicalModelFamilySpec(ModelFamilySpec):
    model_base_class: SupportedModelBaseClass
    expected_batch_type: type[Any]
    expected_batch_contract_kind: ContractKind
    expected_prediction_output_type: type[Any]
    expected_prediction_output_contract_kind: ContractKind


AUTOENCODER_FAMILY = CanonicalModelFamilySpec(
    name="autoencoder",
    model_base_class=BenchRepAutoencoderModel,
    expected_batch_type=AutoencoderBatch,
    expected_batch_contract_kind="typeddict",
    expected_prediction_output_type=AutoencoderPredictionOutput,
    expected_prediction_output_contract_kind="dataclass",
)


VAE_FAMILY = CanonicalModelFamilySpec(
    name="vae",
    model_base_class=BenchRepVAEModel,
    expected_batch_type=AutoencoderBatch,
    expected_batch_contract_kind="typeddict",
    expected_prediction_output_type=VAEPredictionOutput,
    expected_prediction_output_contract_kind="dataclass",
)


COMPOSITE_FAMILY = ModelFamilySpec(
    name="composite",
)


def model_family_supports_reconstruction(
    model_family: ModelFamilySpec,
) -> bool:
    if not isinstance(model_family, CanonicalModelFamilySpec):
        raise TypeError(
            "Static reconstruction support is defined only for canonical "
            "model families."
        )

    prediction_output_type = (
        model_family.expected_prediction_output_type
    )

    if not is_dataclass(prediction_output_type):
        raise TypeError(
            "Reconstruction support detection requires a dataclass "
            "prediction-output contract."
        )

    return any(
        field.name == "reconstruction"
        for field in fields(cast(Any, prediction_output_type))
    )