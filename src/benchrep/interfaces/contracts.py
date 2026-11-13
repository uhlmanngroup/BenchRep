from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Literal, NotRequired
from typing_extensions import TypedDict

import torch

ContractKind: TypeAlias = Literal["typeddict", "dataclass"]
PredictionObsValue: TypeAlias = torch.Tensor | list[int] | list[str]
PredictionMetadata: TypeAlias = dict[str, PredictionObsValue]

# Model input contracts
class AutoencoderBatch(TypedDict):
    """Dictionary batch contract shared by canonical autoencoders and VAEs.

    Attributes
    ----------
    x : torch.Tensor
        Required input tensor to reconstruct.
    sample_id : NotRequired[PredictionObsValue]
        Optional per-sample identifiers.
    label : NotRequired[PredictionObsValue]
        Optional per-sample labels, for annotation and evaluation.
    metadata : NotRequired[PredictionMetadata]
        Optional mapping of additional observation annotations.
    """
    x: torch.Tensor
    sample_id: NotRequired[PredictionObsValue]
    label: NotRequired[PredictionObsValue]
    metadata: NotRequired[PredictionMetadata]


# Model forward contracts
class AutoencoderForwardOutput(TypedDict):
    """Forward-output dictionary produced by the built-in autoencoder.

    Attributes
    ----------
    embedding : torch.Tensor
        Encoded representation.
    reconstruction : torch.Tensor
        Decoder reconstruction.
    """
    embedding: torch.Tensor
    reconstruction: torch.Tensor


class VAEForwardOutput(TypedDict):
    """Forward-output dictionary produced by the built-in VAE.

    Attributes
    ----------
    embedding : torch.Tensor
        Model-agnostic representation; the built-in VAE uses ``z_mu``.
    reconstruction : torch.Tensor
        Decoder reconstruction from the sampled latent variable.
    z_sample : torch.Tensor
        Sample drawn from the approximate posterior and passed to the decoder.
    z_mu : torch.Tensor
        Approximate posterior mean.
    z_logvar : torch.Tensor
        Approximate posterior log variance.
    """

    embedding: torch.Tensor
    reconstruction: torch.Tensor
    z_sample: torch.Tensor
    z_mu: torch.Tensor
    z_logvar: torch.Tensor


# Model prediction contracts
@dataclass(slots=True)
class AutoencoderPredictionOutput:
    """Per-batch prediction contract for an autoencoder-family model.

    Attributes
    ----------
    embedding : torch.Tensor
        Batch of learned representations.
    input : torch.Tensor
        Batch of original model inputs.
    reconstruction : torch.Tensor
        Batch of reconstructed inputs.
    sample_id : PredictionObsValue | None
        Optional per-sample identifiers aligned with the prediction rows.
    label : PredictionObsValue | None
        Optional per-sample labels aligned with the prediction rows.
    metadata : PredictionMetadata | None
        Optional additional observation annotations aligned with the prediction rows.
    """

    # Core representation outputs
    embedding: torch.Tensor

    # Reconstruction-related outputs
    input: torch.Tensor
    reconstruction: torch.Tensor

    # Optional observation-level annotations
    sample_id: PredictionObsValue | None = None
    label: PredictionObsValue | None = None
    metadata: PredictionMetadata | None = None


@dataclass(slots=True)
class VAEPredictionOutput:
    """Per-batch prediction contract for a VAE-family model.

    Attributes
    ----------
    embedding : torch.Tensor
        Batch of representations exported as the model-agnostic embedding; the built-in VAE uses ``z_mu``.
    z_sample : torch.Tensor
        Batch of sampled latent variables.
    z_mu : torch.Tensor
        Batch of approximate posterior means.
    z_logvar : torch.Tensor
        Batch of approximate posterior log variances.
    input : torch.Tensor
        Batch of original model inputs.
    reconstruction : torch.Tensor
        Batch of reconstructed inputs.
    sample_id : PredictionObsValue | None
        Optional per-sample identifiers aligned with the prediction rows.
    label : PredictionObsValue | None
        Optional per-sample labels aligned with the prediction rows.
    metadata : PredictionMetadata | None
        Optional additional observation annotations aligned with the prediction rows.
    """

    # Core representation outputs
    embedding: torch.Tensor
    z_sample: torch.Tensor
    z_mu: torch.Tensor
    z_logvar: torch.Tensor

    # Reconstruction-related outputs
    input: torch.Tensor
    reconstruction: torch.Tensor

    # Optional observation-level annotations
    sample_id: PredictionObsValue | None = None
    label: PredictionObsValue | None = None
    metadata: PredictionMetadata | None = None


@dataclass(slots=True)
class CompositePredictionOutput:
    """Per-batch output of a configuration-built composite model.

    Attributes
    ----------
    model_inputs : dict[str, torch.Tensor]
        Declared input names mapped to their batch tensors.
    model_outputs : dict[str, torch.Tensor]
        Declared output names mapped to their batch tensors.
    batch_metadata : PredictionMetadata
        Observation annotations retained from the batch.
    """

    model_inputs: dict[str, torch.Tensor]
    model_outputs: dict[str, torch.Tensor]
    batch_metadata: PredictionMetadata