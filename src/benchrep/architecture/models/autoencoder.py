from __future__ import annotations

from collections.abc import Callable, Iterable
from typing_extensions import TypedDict, NotRequired

import lightning as L
import torch
from torch import nn

from benchrep.architecture.decoders.base import BaseDecoder
from benchrep.architecture.encoders.base import BaseEncoder
from benchrep.architecture.losses.base import LossTerm


class AutoencoderBatch(TypedDict):
    """Batch contract for autoencoder-style reconstruction models.

    Required:
        x:
            Input tensor to reconstruct.

    Optional:
        sample_id:
            Per-sample identifiers used to track outputs during inference or
            downstream evaluation.
        group:
            Per-sample group/batch/condition labels used for annotation or
            evaluation.
    """
    x: torch.Tensor
    sample_id: NotRequired[torch.Tensor | list[int] | list[str]]
    group: NotRequired[torch.Tensor | list[int] | list[str]]


class AutoencoderOutput(TypedDict):
    """Forward output returned by ``Autoencoder``."""
    embedding: torch.Tensor
    reconstruction: torch.Tensor


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reconstruction_losses: dict[str, LossTerm],
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
    ) -> None:
        super().__init__()

        if encoder.latent_dim != decoder.input_dim:
            raise ValueError(
                f"encoder.latent_dim must match decoder.input_dim, got "
                f"encoder.latent_dim={encoder.latent_dim} and "
                f"decoder.input_dim={decoder.input_dim}."
            )

        if encoder.input_shape is not None and decoder.output_shape is not None:
            if encoder.input_shape != decoder.output_shape:
                raise ValueError(
                    f"encoder.input_shape must match decoder.output_shape, got "
                    f"encoder.input_shape={encoder.input_shape} and "
                    f"decoder.output_shape={decoder.output_shape}."
                )

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_factory = optimizer_factory

        if not reconstruction_losses: # fail fast
            raise ValueError("Autoencoder requires at least one reconstruction loss.")

        for loss_name, weighted_loss in reconstruction_losses.items():
            if weighted_loss.weight < 0:
                raise ValueError(
                    f"Reconstruction loss {loss_name!r} has negative weight "
                    f"{weighted_loss.weight}."
                )

        self.reconstruction_losses = reconstruction_losses

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "decoder",
                "reconstruction_losses",
                "optimizer_factory",
            ]
        )

    def forward(self, x: torch.Tensor) -> AutoencoderOutput:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return {
            "embedding": z,
            "reconstruction": reconstruction,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def training_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="train")

    def validation_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="val")

    def test_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_factory(self.parameters())

    def _compute_loss_step(self, batch: AutoencoderBatch, stage: str) -> torch.Tensor:
        x = self._get_input_from_batch(batch)
        output = self(x)
        reconstruction = output["reconstruction"]
        total_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        # Custom reconstruction losses should inherit from BaseReconstructionLoss,
        # or at least implement forward(reconstruction, target).
        for loss_name, loss_term in self.reconstruction_losses.items():
            raw_loss = loss_term.loss(
                reconstruction=reconstruction,
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
        # BenchRep datamodules/datasets should adapt external data sources into this
        # internal batch contract. For autoencoders, the reconstruction target is
        # the input itself, provided under key 'x'.
        if not isinstance(batch, dict):
            raise TypeError(
                "Autoencoder expects batches to be dictionaries with an 'x' key. "
                f"Got batch type {type(batch).__name__}."
            )

        if "x" not in batch:
            raise KeyError(
                "Autoencoder expected batch to contain key 'x'. "
                f"Available keys: {tuple(batch.keys())}."
            )

        x = batch["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "batch['x'] must be a torch.Tensor, "
                f"got {type(x).__name__}."
            )

        return x