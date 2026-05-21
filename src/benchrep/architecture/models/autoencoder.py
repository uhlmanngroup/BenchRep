from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import lightning as L
import torch
from torch import nn

from benchrep.architecture.decoders.base import BaseDecoder
from benchrep.architecture.encoders.base import BaseEncoder


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reconstruction_loss: nn.Module,
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
        self.reconstruction_loss = reconstruction_loss
        self.optimizer_factory = optimizer_factory

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "decoder",
                "reconstruction_loss",
                "optimizer_factory",
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._run_reconstruction_step(batch, stage="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._run_reconstruction_step(batch, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._run_reconstruction_step(batch, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_factory(self.parameters())

    def _run_reconstruction_step(self, batch: Any, stage: str) -> torch.Tensor:
        x = self._get_input_from_batch(batch)
        reconstruction = self(x)
        loss = self.reconstruction_loss(reconstruction, x)

        self.log(
            f"{stage}/reconstruction_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    @staticmethod
    def _get_input_from_batch(batch: Any) -> torch.Tensor:
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