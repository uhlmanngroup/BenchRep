from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from benchrep.interfaces.models import BenchRepAutoencoderModel
from benchrep.interfaces.contracts import (
    AutoencoderBatch,
    AutoencoderForwardOutput,
    AutoencoderPredictionOutput,
)
from benchrep.architecture.decoders.base import BaseDecoder
from benchrep.architecture.encoders.base import BaseEncoder
from benchrep.architecture.losses.base import LossTerm
from benchrep.architecture.models.utils import validate_loss_weights


class Autoencoder(BenchRepAutoencoderModel):
    """Deterministic autoencoder with weighted role-scoped losses.

    Reconstruction losses receive the reconstructed input and reconstruction
    target. Custom objective losses receive the complete batch and forward
    output, allowing objectives that do not fit the reconstruction-loss
    contract. Terms from both roles contribute additively to the total loss.
    """
    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        reconstruction_losses: dict[str, LossTerm],
        custom_objective_losses: dict[str, LossTerm],
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
    ) -> None:
        super().__init__()

        if encoder.output_dim != decoder.input_dim:
            raise ValueError(
                f"encoder.output_dim must match decoder.input_dim, got "
                f"encoder.output_dim={encoder.output_dim} and "
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

        if not reconstruction_losses and not custom_objective_losses:
            raise ValueError(
                "Autoencoder requires at least one reconstruction loss "
                "or custom objective loss."
            )

        validate_loss_weights(
            {
                "reconstruction": reconstruction_losses,
                "custom_objective": custom_objective_losses,
            }
        )

        self.reconstruction_losses: nn.ModuleDict | None = (
            nn.ModuleDict(reconstruction_losses)
            if reconstruction_losses
            else None
        )
        self.custom_objective_losses: nn.ModuleDict | None = (
            nn.ModuleDict(custom_objective_losses)
            if custom_objective_losses
            else None
        )

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "decoder",
                "reconstruction_losses",
                "custom_objective_losses",
                "optimizer_factory",
            ]
        )

    def forward(self, x: torch.Tensor) -> AutoencoderForwardOutput:
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

    def predict_step(self, batch: AutoencoderBatch, batch_idx: int) -> AutoencoderPredictionOutput:
        x = self._get_input_from_batch(batch)
        output = self(x)

        return AutoencoderPredictionOutput(
            embedding=output["embedding"],
            input=x,
            reconstruction=output["reconstruction"],
            sample_id=batch.get("sample_id"),
            label=batch.get("label"),
            metadata=batch.get("metadata"),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_factory(self.parameters())

    def _compute_loss_step(self, batch: AutoencoderBatch, stage: str) -> torch.Tensor:
        x = self._get_input_from_batch(batch)
        output = self(x)
        reconstruction = output["reconstruction"]
        batch_size = x.shape[0]
        total_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        loss_groups = {
            "reconstruction": (
                self.reconstruction_losses,
                {
                    "reconstruction": reconstruction,
                    "target": x,
                },
            ),
            "custom_objective": (
                self.custom_objective_losses,
                {
                    "batch": batch,
                    "model_output": output,
                },
            ),
        }

        for role, (loss_terms, loss_kwargs) in loss_groups.items():
            if loss_terms is None:
                continue

            role_label = role.replace("_", " ").title()

            for loss_name, loss_term in loss_terms.items():
                raw_loss = loss_term.loss(**loss_kwargs)

                if not isinstance(raw_loss, torch.Tensor):
                    raise TypeError(
                        f"{role_label} loss {loss_name!r} must return a "
                        f"torch.Tensor, got {type(raw_loss).__name__}."
                    )

                if raw_loss.ndim != 0:
                    raise ValueError(
                        f"{role_label} loss {loss_name!r} must return a "
                        f"scalar tensor, got shape {tuple(raw_loss.shape)}."
                    )

                weighted_loss = loss_term.weight * raw_loss
                total_loss = total_loss + weighted_loss

                self.log(
                    f"{stage}/{role}/{loss_name}",
                    raw_loss,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )

                self.log(
                    f"{stage}/{role}/{loss_name}_weighted",
                    weighted_loss,
                    on_step=stage == "train",
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )

        self.log(
            f"{stage}/loss",
            total_loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
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