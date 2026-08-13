from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn

from benchrep.interfaces.models import BenchRepVAEModel
from benchrep.interfaces.contracts import (
    AutoencoderBatch,
    VAEForwardOutput,
    VAEPredictionOutput,
)
from benchrep.architecture.decoders.base import BaseDecoder
from benchrep.architecture.encoders.base import BaseEncoder
from benchrep.architecture.heads.variational import GaussianVariationalHead
from benchrep.architecture.losses.base import LossTerm
from benchrep.architecture.models.utils import validate_loss_weights


class VAE(BenchRepVAEModel):
    """Standard Gaussian variational autoencoder with role-scoped losses.

    The model uses the ``AutoencoderBatch`` contract and follows this structure:

        encoder -> GaussianVariationalHead -> decoder

    The encoder produces deterministic features. The variational head maps those
    features to a diagonal Gaussian posterior and samples a latent vector using the
    reparameterization trick.

    During training, the decoder reconstructs the input from the sampled latent.
    The training objective requires either paired reconstruction and regularization
    losses or at least one custom objective. Custom objectives may be used alone or
    alongside either or both standard loss roles. Reconstruction losses receive the
    reconstruction and target, regularization losses receive the posterior parameters,
    and custom objectives receive the complete batch and forward output. All configured
    terms contribute additively to the total loss.

    During prediction, reconstruction may use either the posterior mean or sampled
    latent, with the posterior mean used by default. The deterministic embedding
    exposed for downstream evaluation is ``z_mu``.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        variational_head: GaussianVariationalHead,
        reconstruction_losses: dict[str, LossTerm],
        regularization_losses: dict[str, LossTerm],
        custom_objective_losses: dict[str, LossTerm],
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
        prediction_reconstruction_latent_source: Literal["mean", "sample"] = "mean",
    ) -> None:
        super().__init__()

        if prediction_reconstruction_latent_source not in {"mean", "sample"}:
            raise ValueError(
                "prediction_reconstruction_latent_source must be 'mean' or 'sample', "
                f"got {prediction_reconstruction_latent_source!r}."
            )

        self.prediction_reconstruction_latent_source = prediction_reconstruction_latent_source

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

        has_custom_objective_losses = bool(custom_objective_losses)
        has_complete_standard_objective = bool(
            reconstruction_losses and regularization_losses
        )

        if not (
                has_custom_objective_losses
                or has_complete_standard_objective
        ):
            raise ValueError(
                "VAE requires at least one custom objective loss or both "
                "reconstruction and regularization losses."
            )

        validate_loss_weights(
            {
                "reconstruction": reconstruction_losses,
                "regularization": regularization_losses,
                "custom_objective": custom_objective_losses,
            }
        )

        self.reconstruction_losses = nn.ModuleDict(reconstruction_losses)
        self.regularization_losses = nn.ModuleDict(regularization_losses)
        self.custom_objective_losses = nn.ModuleDict(custom_objective_losses)

        self.save_hyperparameters(
            ignore=[
                "encoder",
                "decoder",
                "variational_head",
                "reconstruction_losses",
                "regularization_losses",
                "custom_objective_losses",
                "optimizer_factory",
                "prediction_reconstruction_latent_source",
            ]
        )

    def forward(
            self,
            x: torch.Tensor,
            *,
            reconstruction_latent_source: Literal["mean", "sample"] = "sample",
    ) -> VAEForwardOutput:
        encoder_features = self.encode(x)
        latent = self.variational_head(encoder_features)
        if reconstruction_latent_source == "mean":
            reconstruction = self.decode(latent.z_mu)
        elif reconstruction_latent_source == "sample":
            reconstruction = self.decode(latent.z_sample)
        else:
            raise ValueError(
                "reconstruction_latent_source must be 'mean' or 'sample', "
                f"got {reconstruction_latent_source!r}."
            )

        return {
            "embedding": latent.z_mu,
            "reconstruction": reconstruction,
            "z_sample": latent.z_sample,
            "z_mu": latent.z_mu,
            "z_logvar": latent.z_logvar,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def training_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="train")

    def validation_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="val")

    def test_step(self, batch: AutoencoderBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_step(batch, stage="test")

    def predict_step(self, batch: AutoencoderBatch, batch_idx: int) -> VAEPredictionOutput:
        x = self._get_input_from_batch(batch)
        output = self(
            x,
            reconstruction_latent_source=self.prediction_reconstruction_latent_source,
        )

        return VAEPredictionOutput(
            input=x,
            embedding=output["embedding"],
            z_sample=output["z_sample"],
            z_mu=output["z_mu"],
            z_logvar=output["z_logvar"],
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
        z_mu = output["z_mu"]
        z_logvar = output["z_logvar"]
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
            "regularization": (
                self.regularization_losses,
                {
                    "z_mu": z_mu,
                    "z_logvar": z_logvar,
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