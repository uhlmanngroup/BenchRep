from __future__ import annotations

from typing import Any

import lightning as L

from benchrep.architecture.models import Autoencoder
from benchrep.assembly.builders.optimizer_builder import build_optimizer_factory
from benchrep.assembly.config_utils import (
    get_optional_section,
    get_required_section,
    get_required_value,
    normalize_name,
    require_mapping,
)
from benchrep.assembly.registry import (
    DECODERS,
    ENCODERS,
    MODELS,
    RECONSTRUCTION_LOSSES,
)


def build_model(config: dict[str, Any]) -> L.LightningModule:
    """Build a model from config.

    This is the public model-builder entry point. It reads ``config["model"]["name"]``
    and dispatches to the matching model-specific builder.

    Each model-specific builder is responsible for requiring only the config
    sections that its model type actually needs. For example, an autoencoder
    requires an encoder, decoder, reconstruction loss, and optimizer, while a
    future contrastive model may require an encoder, projection head, contrastive
    loss, and optimizer, but no decoder.

    Parameters
    ----------
    config:
        Full loaded config dictionary.

    Returns
    -------
    L.LightningModule
        Instantiated Lightning model ready to be passed to a Lightning Trainer.
    """
    config = require_mapping(config, "config")
    model_config = get_required_section(config, "model")

    model_name = normalize_name(
        get_required_value(model_config, "name"),
        field_name="config['model']['name']",
    )

    if model_name == "autoencoder":
        return _build_autoencoder(config)

    raise ValueError(
        f"Unsupported model name {model_name!r}. "
        f"Available options: {MODELS.keys()}."
    )


def _build_autoencoder(config: dict[str, Any]) -> Autoencoder:
    # Autoencoders require encoder, decoder, reconstruction loss, and optimizer
    # sections. Other model types should define their own model-specific builders.
    encoder_config = get_required_section(config, "encoder")
    decoder_config = get_required_section(config, "decoder")
    loss_config = get_required_section(config, "loss")
    optimizer_config = get_required_section(config, "optimizer")

    encoder = _build_encoder(encoder_config)
    decoder = _build_decoder(
        decoder_config,
        latent_dim=encoder.latent_dim,
    )
    reconstruction_loss = _build_reconstruction_loss(loss_config)
    optimizer_factory = build_optimizer_factory(optimizer_config)

    model_class = MODELS.get("autoencoder")

    return model_class(
        encoder=encoder,
        decoder=decoder,
        reconstruction_loss=reconstruction_loss,
        optimizer_factory=optimizer_factory,
    )


def _build_encoder(encoder_config: dict[str, Any]) -> Any:
    encoder_config = require_mapping(encoder_config, "encoder_config")

    encoder_name = normalize_name(
        get_required_value(encoder_config, "name"),
        field_name="config['encoder']['name']",
    )
    encoder_params = get_optional_section(encoder_config, "params")

    return ENCODERS.create(encoder_name, **encoder_params)


def _build_decoder(decoder_config: dict[str, Any], latent_dim: int) -> Any:
    """Build a decoder and wire its latent input dimension from the encoder."""
    decoder_config = require_mapping(decoder_config, "decoder_config")

    decoder_name = normalize_name(
        get_required_value(decoder_config, "name"),
        field_name="config['decoder']['name']",
    )
    decoder_params = dict(get_optional_section(decoder_config, "params"))

    # Wire decoder input dimensionality from the encoder latent representation.
    decoder_params["latent_dim"] = latent_dim

    return DECODERS.create(decoder_name, **decoder_params)


def _build_reconstruction_loss(loss_config: dict[str, Any]) -> Any:
    loss_config = require_mapping(loss_config, "loss_config")
    reconstruction_config = get_required_section(loss_config, "reconstruction")

    loss_name = normalize_name(
        get_required_value(reconstruction_config, "name"),
        field_name="config['loss']['reconstruction']['name']",
    )
    loss_params = get_optional_section(reconstruction_config, "params")

    return RECONSTRUCTION_LOSSES.create(loss_name, **loss_params)