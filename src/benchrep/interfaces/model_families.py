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
    """Model-family identifier used by workflow configuration interfaces.

    Attributes
    ----------
    name : str
        Canonical model-family name.

    Notes
    -----
    Use the predefined family constants. Constructing a spec does not register a new model family.
    """
    name: str


@dataclass(frozen=True)
class CanonicalModelFamilySpec(ModelFamilySpec):
    """Canonical-family contract extending the model-family name.

    Attributes
    ----------
    model_base_class : SupportedModelBaseClass
        Required base class for a whole-model runtime override.
    expected_batch_type : type[Any]
        Expected canonical batch dictionary type.
    expected_batch_contract_kind : ContractKind
        Representation of the batch contract; the canonical families use ``typeddict``.
    expected_prediction_output_type : type[Any]
        Expected per-batch prediction output type.
    expected_prediction_output_contract_kind : ContractKind
        Representation of the prediction contract; the canonical families use ``dataclass``.

    Notes
    -----
    The inherited ``name`` identifies the family. Prefer ``AUTOENCODER_FAMILY`` or ``VAE_FAMILY``.
    """
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
