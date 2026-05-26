from __future__ import annotations

from typing import Any

import lightning as L

from benchrep.architecture.models import Autoencoder
from benchrep.assembly.builders.optimizer_builder import build_optimizer_factory
from benchrep.assembly.config_utils import normalize_name
from benchrep.assembly.schemas import (
    BenchRepConfig,
    DecoderConfig,
    EncoderConfig,
    LossConfig,
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
        return _build_autoencoder(config)

    raise ValueError(
        f"Unsupported model name {model_name!r}. "
        f"Available options: {MODELS.keys()}."
    )


def _build_autoencoder(config: BenchRepConfig) -> Autoencoder:
    # Autoencoders require encoder, decoder, reconstruction loss, and optimizer
    # sections. Other model types should define their own model-specific builders.
    if config.decoder is None:
        raise ValueError("Autoencoder model requires a decoder config section.")

    try:
        reconstruction_loss_config = config.losses["reconstruction"]
    except KeyError as error:
        raise ValueError(
            "Autoencoder requires a reconstruction loss under "
            "`losses.reconstruction`."
        ) from error

    encoder = _build_encoder(config.encoder)
    decoder = _build_decoder(
        config.decoder,
        latent_dim=encoder.latent_dim,
    )
    optimizer_factory = build_optimizer_factory(config.optimizer)
    reconstruction_loss = _build_reconstruction_loss(reconstruction_loss_config)
    model_class = MODELS.get("autoencoder")

    return model_class(
        encoder=encoder,
        decoder=decoder,
        reconstruction_loss=reconstruction_loss,
        optimizer_factory=optimizer_factory,
    )


def _build_encoder(encoder_config: EncoderConfig) -> Any:
    encoder_name = normalize_name(
        encoder_config.name,
        field_name="config.encoder.name",
    )

    return ENCODERS.create(encoder_name, **encoder_config.params)


def _build_decoder(decoder_config: DecoderConfig, latent_dim: int) -> Any:
    """Build a decoder and wire its latent input dimension from the encoder."""
    decoder_name = normalize_name(
        decoder_config.name,
        field_name="config.decoder.name",
    )
    decoder_params = dict(decoder_config.params)

    # Wire decoder input dimensionality from the encoder latent representation.
    decoder_params["latent_dim"] = latent_dim

    return DECODERS.create(decoder_name, **decoder_params)


def _build_reconstruction_loss(loss_config: LossConfig) -> Any:
    loss_name = normalize_name(
        loss_config.name,
        field_name="config.losses.reconstruction.name",
    )

    return RECONSTRUCTION_LOSSES.create(loss_name, **loss_config.params)