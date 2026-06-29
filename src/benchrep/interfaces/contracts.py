from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, NotRequired
from typing_extensions import TypedDict

import torch


PredictionObsValue: TypeAlias = torch.Tensor | list[int] | list[str]
PredictionMetadata: TypeAlias = dict[str, PredictionObsValue]

# Model input contracts
class AutoencoderBatch(TypedDict):
    """Batch contract for autoencoder-style reconstruction models.

    Required:
        x:
            Input tensor to reconstruct.

    Optional:
        sample_id:
            Per-sample identifiers used to track outputs during inference or
            downstream evaluation.
        label:
            Per-sample labels used for annotation or
            evaluation (e.g. cell type).
        metadata:
            Additional per-sample metadata, e.g. patient_id, batch,
            treatment_group, or timepoint.
    """
    x: torch.Tensor
    sample_id: NotRequired[PredictionObsValue]
    label: NotRequired[PredictionObsValue]
    metadata: NotRequired[PredictionMetadata]


# Model forward contracts
class AutoencoderForwardOutput(TypedDict):
    """Forward output returned by ``Autoencoder``."""
    embedding: torch.Tensor
    reconstruction: torch.Tensor


class VAEForwardOutput(TypedDict):
    """
    Forward output returned by ``VAE``.

    `embedding` is the model-agnostic representation used by generic BenchRep
    inference/evaluation code. For VAEs, it is intentionally set to `z_mu`
    (i.e. redundant by design).

    `z_mu` and `z_logvar` are the approximate posterior parameters used for
    KL divergence. `z_sample` is the stochastic latent sample used by the decoder.
    """

    embedding: torch.Tensor
    reconstruction: torch.Tensor
    z_sample: torch.Tensor
    z_mu: torch.Tensor
    z_logvar: torch.Tensor


# Model prediction contracts
@dataclass(slots=True)
class AutoencoderPredictionOutput:
    """Prediction output returned by autoencoder-family models."""

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
    """Prediction output returned by VAE-family models."""

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
