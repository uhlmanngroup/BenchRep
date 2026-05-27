from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

import lightning as L

from benchrep.architecture.encoders import BaseEncoder
from benchrep.architecture.decoders import BaseDecoder
from benchrep.architecture.models import Autoencoder
from benchrep.architecture.losses.base import LossTerm
from benchrep.assembly.builders.optimizer_builder import build_optimizer_factory
from benchrep.assembly.config_utils import normalize_name
from benchrep.assembly.schemas import (
    BenchRepConfig,
    DecoderConfig,
    EncoderConfig,
    LossesConfig,
    OptimizerConfig,
)
from benchrep.assembly.registry import (
    DECODERS,
    ENCODERS,
    MODELS,
    RECONSTRUCTION_LOSSES,
)


def build_model(config: BenchRepConfig) -> L.LightningModule:
    """Build a model from config.

    This is the public model-builder entry point. It reads ``config.model.name``
    and dispatches to the matching model-specific builder.

    Each model-specific builder is responsible for requiring only the config
    sections that its model type actually needs. For example, an autoencoder
    requires an encoder, decoder, reconstruction loss, and optimizer, while a
    future contrastive model may require an encoder, projection head, contrastive
    loss, and optimizer, but no decoder.

    Parameters
    ----------
    config:
        Validated BenchRep config object.

    Returns
    -------
    L.LightningModule
        Instantiated Lightning model ready to be passed to a Lightning Trainer.
    """
    model_name = normalize_name(
        config.model.name,
        field_name="config.model.name",
    )

    if model_name == "autoencoder":
        if config.decoder is None:
            raise ValueError("Autoencoder requires a decoder config section.")

        return build_autoencoder(
            encoder=config.encoder,
            decoder=config.decoder,
            optimizer=config.optimizer,
            reconstruction_losses=config.losses["reconstruction"],
        )

    raise ValueError(
        f"Unsupported model name {model_name!r}. "
        f"Available options: {MODELS.keys()}."
    )


def build_autoencoder(
    encoder: EncoderConfig | BaseEncoder,
    decoder: DecoderConfig | BaseDecoder,
    optimizer: (
            OptimizerConfig |
            Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
    ),
    reconstruction_losses: dict[str, LossesConfig | LossTerm],
) -> Autoencoder:
    # Resolve configs objs into instantiated components where needed
    if isinstance(encoder, EncoderConfig):
        encoder = _build_encoder(encoder)

    if isinstance(decoder, DecoderConfig):
        decoder = _build_decoder(decoder, latent_dim=encoder.latent_dim)

    if isinstance(optimizer, OptimizerConfig):
        optimizer_factory = build_optimizer_factory(optimizer)
    else:
        optimizer_factory = optimizer

    reconstruction_losses = _build_reconstruction_losses(reconstruction_losses)

    model_class = MODELS.get("autoencoder")

    return model_class(
        encoder=encoder,
        decoder=decoder,
        reconstruction_losses=reconstruction_losses,
        optimizer_factory=optimizer_factory,
    )


def _build_encoder(encoder_config: EncoderConfig) -> BaseEncoder:
    encoder_name = normalize_name(
        encoder_config.name,
        field_name="config.encoder.name",
    )

    return ENCODERS.create(encoder_name, **encoder_config.params)


def _build_decoder(decoder_config: DecoderConfig, latent_dim: int) -> BaseDecoder:
    """Build a decoder and wire its latent input dimension from the encoder."""
    decoder_name = normalize_name(
        decoder_config.name,
        field_name="config.decoder.name",
    )
    decoder_params = dict(decoder_config.params)

    # Wire decoder input dimensionality from the encoder latent representation.
    decoder_params["latent_dim"] = latent_dim

    return DECODERS.create(decoder_name, **decoder_params)


def _build_reconstruction_losses(
    reconstruction_losses: dict[str, LossesConfig | LossTerm],
) -> dict[str, LossTerm]:
    loss_terms: dict[str, LossTerm] = {}

    for loss_name, loss_spec in reconstruction_losses.items():
        if isinstance(loss_spec, LossTerm):
            loss_terms[loss_name] = loss_spec
            continue

        loss_terms[loss_name] = LossTerm(
            loss=RECONSTRUCTION_LOSSES.create(loss_name, **loss_spec.params),
            weight=loss_spec.weight,
        )

    return loss_terms


