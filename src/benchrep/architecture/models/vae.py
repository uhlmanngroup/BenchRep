from __future__ import annotations

from collections.abc import Callable, Iterable

from dataclasses import dataclass, field
from typing import TypeAlias
from typing_extensions import TypedDict, NotRequired

import lightning as L
import torch
from torch import nn

from benchrep.architecture.models.autoencoder import AutoencoderBatch
from benchrep.architecture.decoders.base import BaseDecoder
from benchrep.architecture.encoders.base import BaseEncoder
from benchrep.architecture.heads.variational import GaussianVariationalHead
from benchrep.architecture.losses.base import LossTerm


PredictionObsValue: TypeAlias = torch.Tensor | list[int] | list[str]
PredictionMetadata: TypeAlias = dict[str, PredictionObsValue]


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

@dataclass(slots=True)
class VAEPredictionOutput:
    """Prediction output returned by VAE-family models."""

    # Core representation outputs
    embedding: torch.Tensor
    z_sample: torch.Tensor
    z_mu: torch.Tensor
    z_logvar: torch.Tensor

    # Reconstruction-related outputs
    input: torch.Tensor | None = None
    reconstruction: torch.Tensor | None = None

    # Optional observation-level annotations
    sample_id: PredictionObsValue | None = None
    label: PredictionObsValue | None = None
    metadata: PredictionMetadata = field(default_factory=dict)


class VAE(L.LightningModule):
    """Standard Gaussian variational autoencoder.

    The model follows the standard VAE structure and uses the Autoencoder
    batch contract, ``AutoencoderBatch``:

        encoder -> GaussianVariationalHead -> decoder

    The encoder produces deterministic features. The variational head maps those
    features to a diagonal Gaussian posterior, samples a latent vector with the
    reparameterization trick, and the decoder reconstructs the input from the
    sampled latent vector.

    The deterministic embedding exposed for downstream evaluation is ``z_mu``.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        variational_head: GaussianVariationalHead,
        reconstruction_losses: dict[str, LossTerm],
        regularization_losses: dict[str, LossTerm],
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
    ) -> None:
        super().__init__()

        if encoder.input_shape is not None and decoder.output_shape is not None:
            if encoder.input_shape != decoder.output_shape:
                raise ValueError(
                    f"encoder.input_shape must match decoder.output_shape, got "
                    f"encoder.input_shape={encoder.input_shape} and "
                    f"decoder.output_shape={decoder.output_shape}."
                )

        if decoder.input_dim != variational_head.latent_dim:
            raise ValueError(
                f"decoder.input_dim must match variational_head.latent_dim, got "
                f"decoder.input_dim={decoder.input_dim} and "
                f"variational_head.latent_dim={variational_head.latent_dim}."
            )

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_factory = optimizer_factory

        self.variational_head = variational_head

        if not reconstruction_losses:  # fail fast
            raise ValueError("VAE requires at least one reconstruction loss.")

        if not regularization_losses:  # fail fast
            raise ValueError("VAE requires at least one regularization loss.")

        for loss_name, loss_term in reconstruction_losses.items():
            if loss_term.weight < 0:
                raise ValueError(
                    f"Reconstruction loss {loss_name!r} has negative weight "
                    f"{loss_term.weight}."
                )

        for loss_name, loss_term in regularization_losses.items():
            if loss_term.weight < 0:
                raise ValueError(
                    f"Regularization loss {loss_name!r} has negative weight "
                    f"{loss_term.weight}."
                )

        self.reconstruction_losses = reconstruction_losses
        self.regularization_losses = regularization_losses

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "decoder",
                "variational_head",
                "reconstruction_losses",
                "regularization_losses",
                "optimizer_factory",
            ]
        )

    def forward(self, x: torch.Tensor) -> VAEForwardOutput:
        encoder_features = self.encode(x)
        latent = self.variational_head(encoder_features)
        reconstruction = self.decode(latent.z_sample)

        return {
            "embedding": latent.z_mu,
            "reconstruction": reconstruction,
            "z_sample": latent.z_sample,
            "z_mu": latent.z_mu,
            "z_logvar": latent.z_logvar,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_sample: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_sample)

    def training_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="train")

    def validation_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="val")

    def test_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="test")

    def predict_step(self, batch: AutoencoderBatch, batch_idx: int) -> VAEPredictionOutput:
        x = self._get_input_from_batch(batch)
        output = self(x)

        return VAEPredictionOutput(
            input=x,
            embedding=output["embedding"],
            z_sample=output["z_sample"],
            z_mu=output["z_mu"],
            z_logvar=output["z_logvar"],
            reconstruction=output["reconstruction"],
            sample_id=batch.get("sample_id"),
            label=batch.get("label"),
            metadata=batch.get("metadata", {}),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_factory(self.parameters())

    def _compute_loss_step(self, batch: AutoencoderBatch, stage: str) -> torch.Tensor:
        x = self._get_input_from_batch(batch)
        output = self(x)

        total_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        # The VAE owns tensor routing for reconstruction losses. Anything
        # registered under losses.reconstruction must follow the
        # BaseReconstructionLoss interface: forward(reconstruction, target).
        for loss_name, loss_term in self.reconstruction_losses.items():
            raw_loss = loss_term.loss(
                reconstruction=output["reconstruction"],
                target=x,
            )
            weighted_loss_value = loss_term.weight * raw_loss
            total_loss = total_loss + weighted_loss_value

            self.log(
                f"{stage}/reconstruction/{loss_name}",
                raw_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
            )

            self.log(
                f"{stage}/reconstruction/{loss_name}_weighted",
                weighted_loss_value,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
            )

        # Standard Gaussian VAE regularization losses should follow the
        # GaussianKLDivergenceLoss interface: forward(z_mu, z_logvar).
        for loss_name, loss_term in self.regularization_losses.items():
            raw_loss = loss_term.loss(
                z_mu=output["z_mu"],
                z_logvar=output["z_logvar"],
            )
            weighted_loss_value = loss_term.weight * raw_loss
            total_loss = total_loss + weighted_loss_value

            self.log(
                f"{stage}/regularization/{loss_name}",
                raw_loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
            )

            self.log(
                f"{stage}/regularization/{loss_name}_weighted",
                weighted_loss_value,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=False,
            )

        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss

    @staticmethod
    def _get_input_from_batch(batch: AutoencoderBatch) -> torch.Tensor:
        # BenchRep datamodules/datasets should adapt external data sources into
        # this internal batch contract. For VAEs, the reconstruction target is
        # the input itself, provided under key 'x'.
        if not isinstance(batch, dict):
            raise TypeError(
                "VAE expects batches to be dictionaries with an 'x' key. "
                f"Got batch type {type(batch).__name__}."
            )

        if "x" not in batch:
            raise KeyError(
                "VAE expected batch to contain key 'x'. "
                f"Available keys: {tuple(batch.keys())}."
            )

        x = batch["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "batch['x'] must be a torch.Tensor, "
                f"got {type(x).__name__}."
            )

        return x