from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

import lightning as L

from benchrep.records import get_run_logger
from benchrep.architecture.encoders import BaseEncoder
from benchrep.architecture.decoders import BaseDecoder
from benchrep.architecture.heads import GaussianVariationalHead
from benchrep.architecture.models import (
    Autoencoder,
    VAE,
)
from benchrep.architecture.losses.base import LossTerm
from benchrep.assembly.builders.optimizer_builder import build_optimizer_factory
from benchrep.assembly.config_utils import normalize_name
from benchrep.assembly.schemas import (
    BenchRepConfig,
    DecoderConfig,
    EncoderConfig,
    LossTermConfig,
    OptimizerConfig,
)
from benchrep.assembly.registry import (
    DECODERS,
    ENCODERS,
    MODELS,
    OPTIMIZERS,
    RECONSTRUCTION_LOSSES,
    REGULARIZATION_LOSSES,
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
    run_log = get_run_logger()

    model_name = normalize_name(
        config.model.name,
        field_name="config.model.name",
    )

    run_log.info("Building model components...")

    model_cls = MODELS.get(model_name)

    if model_cls is Autoencoder:
        if config.decoder is None:
            raise ValueError("Autoencoder requires a decoder config section.")

        model = build_autoencoder(
            encoder=config.encoder,
            decoder=config.decoder,
            optimizer=config.optimizer,
            reconstruction_losses=config.losses["reconstruction"],
        )

        run_log.info("Assembled model: %s", type(model).__name__)

        return model

    elif model_cls is VAE:
        if config.decoder is None:
            raise ValueError("VAE requires a decoder config section.")

        model = build_vae(
            encoder=config.encoder,
            decoder=config.decoder,
            optimizer=config.optimizer,
            latent_dim=config.model.params["latent_dim"],
            reconstruction_losses=config.losses["reconstruction"],
            regularization_losses=config.losses["regularization"],
        )

        run_log.info("Assembled model: %s", type(model).__name__)

        return model

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
    reconstruction_losses: dict[str, LossTermConfig | LossTerm],
) -> Autoencoder:
    run_log = get_run_logger()

    # Resolve configs objs into instantiated components where needed
    if isinstance(encoder, EncoderConfig):
        encoder_name = encoder.name
        encoder = _build_encoder(encoder)
        run_log.info("Built encoder from config: %s -> %s",
                     encoder_name,
                     type(encoder).__name__)

    if isinstance(decoder, DecoderConfig):
        decoder_name = decoder.name
        decoder = _build_decoder(decoder, input_dim=encoder.output_dim)
        run_log.info("Built decoder from config: %s -> %s",
                     decoder_name,
                     type(decoder).__name__)

    if isinstance(optimizer, OptimizerConfig):
        optimizer_name = optimizer.name
        optimizer_cls = OPTIMIZERS.get(optimizer_name) # Use registry as optimizer is built as a factory
        optimizer_factory = build_optimizer_factory(optimizer)
        run_log.info("Built optimizer factory from config: %s -> %s",
                     optimizer_name,
                     optimizer_cls.__name__)
    else:
        optimizer_factory = optimizer

    loss_sources = {
        loss_name: "pre-built" if isinstance(loss_spec, LossTerm) else "config"
        for loss_name, loss_spec in reconstruction_losses.items()
    }

    reconstruction_losses = _build_reconstruction_losses(reconstruction_losses)
    run_log.info(
        "Resolved reconstruction losses: %s",
        ", ".join(
            (
                f"{loss_name} ({loss_sources[loss_name]})"
                f" -> {type(loss_term.loss).__name__}"
                f" (weight={loss_term.weight})"
            )
            for loss_name, loss_term in reconstruction_losses.items()
        ),
    )

    return Autoencoder(
        encoder=encoder,
        decoder=decoder,
        reconstruction_losses=reconstruction_losses,
        optimizer_factory=optimizer_factory,
    )


def build_vae(
    encoder: EncoderConfig | BaseEncoder,
    decoder: DecoderConfig | BaseDecoder,
    optimizer: (
            OptimizerConfig |
            Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
    ),
    latent_dim: int,
    reconstruction_losses: dict[str, LossTermConfig | LossTerm],
    regularization_losses: dict[str, LossTermConfig | LossTerm],
) -> VAE:
    run_log = get_run_logger()

    # Resolve configs objs into instantiated components where needed
    if isinstance(encoder, EncoderConfig):
        encoder_name = encoder.name
        encoder = _build_encoder(encoder)
        run_log.info("Built encoder from config: %s -> %s",
                     encoder_name,
                     type(encoder).__name__)

    if isinstance(decoder, DecoderConfig):
        decoder_name = decoder.name
        decoder = _build_decoder(decoder, input_dim=latent_dim)
        run_log.info("Built decoder from config: %s -> %s",
                     decoder_name,
                     type(decoder).__name__)

    if isinstance(optimizer, OptimizerConfig):
        optimizer_name = optimizer.name
        optimizer_cls = OPTIMIZERS.get(optimizer_name) # Use registry as optimizer is built as a factory
        optimizer_factory = build_optimizer_factory(optimizer)
        run_log.info("Built optimizer factory from config: %s -> %s",
                     optimizer_name,
                     optimizer_cls.__name__)
    else:
        optimizer_factory = optimizer

    reconstruction_loss_sources = {
        loss_name: "pre-built" if isinstance(loss_spec, LossTerm) else "config"
        for loss_name, loss_spec in reconstruction_losses.items()
    }

    variational_head = GaussianVariationalHead(
        in_features=encoder.output_dim,
        latent_dim=latent_dim,
    )

    reconstruction_losses = _build_reconstruction_losses(reconstruction_losses)
    run_log.info(
        "Resolved reconstruction losses: %s",
        ", ".join(
            (
                f"{loss_name} ({reconstruction_loss_sources[loss_name]})"
                f" -> {type(loss_term.loss).__name__}"
                f" (weight={loss_term.weight})"
            )
            for loss_name, loss_term in reconstruction_losses.items()
        ),
    )

    regularization_loss_sources = {
        loss_name: "pre-built" if isinstance(loss_spec, LossTerm) else "config"
        for loss_name, loss_spec in regularization_losses.items()
    }

    regularization_losses = _build_regularization_losses(regularization_losses)
    run_log.info(
        "Resolved regularization losses: %s",
        ", ".join(
            (
                f"{loss_name} ({regularization_loss_sources[loss_name]})"
                f" -> {type(loss_term.loss).__name__}"
                f" (weight={loss_term.weight})"
            )
            for loss_name, loss_term in regularization_losses.items()
        ),
    )

    return VAE(
        encoder=encoder,
        decoder=decoder,
        variational_head=variational_head,
        reconstruction_losses=reconstruction_losses,
        regularization_losses=regularization_losses,
        optimizer_factory=optimizer_factory,
    )


def _build_encoder(encoder_config: EncoderConfig) -> BaseEncoder:
    encoder_name = normalize_name(
        encoder_config.name,
        field_name="config.encoder.name",
    )

    return ENCODERS.create(encoder_name, **encoder_config.params)


def _build_decoder(decoder_config: DecoderConfig, input_dim: int) -> BaseDecoder:
    """Build a decoder and set its expected input dimensionality."""
    decoder_name = normalize_name(
        decoder_config.name,
        field_name="config.decoder.name",
    )
    decoder_params = dict(decoder_config.params)

    # Wire decoder input dimensionality from the supplied input dimension.
    decoder_params["input_dim"] = input_dim

    return DECODERS.create(decoder_name, **decoder_params)


def _build_reconstruction_losses(
    reconstruction_losses: dict[str, LossTermConfig | LossTerm],
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


def _build_regularization_losses(
    regularization_losses: dict[str, LossTermConfig | LossTerm],
) -> dict[str, LossTerm]:
    loss_terms: dict[str, LossTerm] = {}

    for loss_name, loss_spec in regularization_losses.items():
        if isinstance(loss_spec, LossTerm):
            loss_terms[loss_name] = loss_spec
            continue

        loss_terms[loss_name] = LossTerm(
            loss=REGULARIZATION_LOSSES.create(loss_name, **loss_spec.params),
            weight=loss_spec.weight,
        )

    return loss_terms