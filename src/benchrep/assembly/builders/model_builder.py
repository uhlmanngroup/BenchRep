from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn

from benchrep.assembly.builders.architecture_builder import build_encoder, build_decoder
from benchrep.records import get_run_logger
from benchrep.architecture.encoders import BaseEncoder
from benchrep.architecture.decoders import BaseDecoder
from benchrep.architecture.heads import GaussianVariationalHead
from benchrep.architecture.models import (
    Autoencoder,
    VAE,
)
from benchrep.assembly.builders.loss_builder import build_loss_terms
from benchrep.architecture.losses.base import LossTerm
from benchrep.assembly.builders.optimizer_builder import build_optimizer_factory
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.schemas import (
    TrainingConfig,
    TrainingDecoderConfig,
    TrainingEncoderConfig,
    TrainingLossTermConfig,
    TrainingOptimizerConfig,
)
from benchrep.assembly.registries.core import (
    MODELS,
    OPTIMIZERS,
)
from benchrep.interfaces.model_families import SupportedModel


def build_model(
    config: TrainingConfig,
    *,
    prediction_reconstruction_latent_source: (
        Literal["mean", "sample"] | None
    ) = None,
) -> SupportedModel:
    """Build a model from config.

    This is the public model-builder entry point. It reads ``config.model.name``
    and dispatches to the matching model-specific builder.

    Each model-specific builder is responsible for requiring only the config
    sections that its model type actually needs. For example, an autoencoder
    requires an encoder, decoder, compatible loss configuration, and optimizer,
    while a future contrastive model may require an encoder, projection head,
    contrastive loss, and optimizer, but no decoder.

    Parameters
    ----------
    config:
        Validated BenchRep config object.
    prediction_reconstruction_latent_source:
        VAE-only selection of the latent representation decoded during
        prediction. ``"mean"`` uses the posterior mean and ``"sample"`` uses
        the sampled latent. ``None`` resolves to ``"mean"`` for VAEs and is
        required for autoencoders.

    Returns
    -------
    SupportedModel
        Instantiated BenchRep autoencoder or variational autoencoder.
    """
    run_log = get_run_logger()

    if config.model is None:
        raise ValueError("Model config section is required.")

    model_name = normalize_name(
        config.model.name,
        field_name="config.model.name",
    )

    run_log.info("Building model components...")

    model_cls = MODELS.get(model_name)

    if model_cls is Autoencoder:
        if config.encoder is None:
            raise ValueError("Autoencoder requires an encoder config section.")
        if config.decoder is None:
            raise ValueError("Autoencoder requires a decoder config section.")
        if config.optimizer is None:
            raise ValueError("Autoencoder requires an optimizer config section.")
        if prediction_reconstruction_latent_source is not None:
            raise ValueError(
                "`prediction_reconstruction_latent_source` is only supported "
                "when building a VAE."
            )
        if config.losses is None:
            raise ValueError("Autoencoder requires a losses config section.")

        model = build_autoencoder(
            encoder=config.encoder,
            decoder=config.decoder,
            optimizer=config.optimizer,
            reconstruction_losses=config.losses.get(
                "reconstruction",
                {},
            ),
            custom_objective_losses=config.losses.get(
                "custom_objective",
                {},
            ),
        )

        run_log.info("Assembled model: %s", type(model).__name__)

        return model

    elif model_cls is VAE:
        if config.encoder is None:
            raise ValueError("VAE requires an encoder config section.")
        if config.decoder is None:
            raise ValueError("VAE requires a decoder config section.")
        if config.optimizer is None:
            raise ValueError("VAE requires an optimizer config section.")
        if config.model.params is None:
            raise ValueError("VAE requires `model.params`.")
        if config.losses is None:
            raise ValueError("VAE requires a losses config section.")

        latent_dim = config.model.params.get("latent_dim")
        if (
                not isinstance(latent_dim, int)
                or isinstance(latent_dim, bool)
                or latent_dim <= 0
        ):
            raise ValueError(
                "VAE requires `model.params.latent_dim` to be a positive integer."
            )


        if prediction_reconstruction_latent_source is None:
            prediction_reconstruction_latent_source: Literal["mean", "sample"] = "mean"

        model = build_vae(
            encoder=config.encoder,
            decoder=config.decoder,
            optimizer=config.optimizer,
            latent_dim=latent_dim,
            reconstruction_losses=config.losses.get(
                "reconstruction",
                {},
            ),
            regularization_losses=config.losses.get(
                "regularization",
                {},
            ),
            custom_objective_losses=config.losses.get(
                "custom_objective",
                {},
            ),
            prediction_reconstruction_latent_source=(
                prediction_reconstruction_latent_source
            ),
        )

        run_log.info("Assembled model: %s", type(model).__name__)

        return model

    raise ValueError(
        f"Unsupported model name {model_name!r}. "
        f"Available options: {MODELS.keys()}."
    )


def build_autoencoder(
    encoder: TrainingEncoderConfig | BaseEncoder,
    decoder: TrainingDecoderConfig | BaseDecoder,
    optimizer: (
            TrainingOptimizerConfig |
            Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
    ),
    reconstruction_losses: dict[str, TrainingLossTermConfig | LossTerm],
    custom_objective_losses: dict[
        str,
        TrainingLossTermConfig | LossTerm,
    ],
) -> Autoencoder:
    run_log = get_run_logger()

    # Resolve configs objs into instantiated components where needed
    if isinstance(encoder, TrainingEncoderConfig):
        encoder_name = encoder.name
        encoder = build_encoder(encoder)
        run_log.info("Built encoder from config: %s -> %s",
                     encoder_name,
                     type(encoder).__name__)
    else:
        run_log.info("Using provided encoder: %s",
                     type(encoder).__name__)

    if isinstance(decoder, TrainingDecoderConfig):
        decoder_name = decoder.name
        decoder = build_decoder(
            decoder,
            input_dim=encoder.output_dim,
            encoder=encoder,
        )
        run_log.info("Built decoder from config: %s -> %s",
                     decoder_name,
                     type(decoder).__name__)
    else:
        run_log.info("Using provided decoder: %s",
                     type(decoder).__name__)

    if isinstance(optimizer, TrainingOptimizerConfig):
        optimizer_name = optimizer.name
        optimizer_cls = OPTIMIZERS.get(optimizer_name) # Use registry as optimizer is built as a factory
        optimizer_factory = build_optimizer_factory(optimizer)
        run_log.info("Built optimizer factory from config: %s -> %s",
                     optimizer_name,
                     optimizer_cls.__name__)
    else:
        optimizer_factory = optimizer
        run_log.info("Using provided optimizer factory: %s",
                     getattr(optimizer, "__name__", type(optimizer).__name__))

    losses_by_role = build_loss_terms(
        {
            "reconstruction": reconstruction_losses,
            "custom_objective": custom_objective_losses,
        },
    )

    return Autoencoder(
        encoder=encoder,
        decoder=decoder,
        reconstruction_losses=losses_by_role["reconstruction"],
        custom_objective_losses=losses_by_role["custom_objective"],
        optimizer_factory=optimizer_factory,
    )


def build_vae(
    encoder: TrainingEncoderConfig | BaseEncoder,
    decoder: TrainingDecoderConfig | BaseDecoder,
    optimizer: (
            TrainingOptimizerConfig |
            Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
    ),
    latent_dim: int,
    reconstruction_losses: dict[str, TrainingLossTermConfig | LossTerm],
    regularization_losses: dict[str, TrainingLossTermConfig | LossTerm],
    custom_objective_losses: dict[
        str,
        TrainingLossTermConfig | LossTerm,
    ],
    prediction_reconstruction_latent_source: (
            Literal["mean", "sample"]
    ) = "mean",
) -> VAE:
    run_log = get_run_logger()

    # Resolve configs objs into instantiated components where needed
    if isinstance(encoder, TrainingEncoderConfig):
        encoder_name = encoder.name
        encoder = build_encoder(encoder)
        run_log.info("Built encoder from config: %s -> %s",
                     encoder_name,
                     type(encoder).__name__)
    else:
        run_log.info("Using provided encoder: %s",
                     type(encoder).__name__)

    if isinstance(decoder, TrainingDecoderConfig):
        decoder_name = decoder.name
        decoder = build_decoder(
            decoder,
            input_dim=latent_dim,
            encoder=encoder,
        )
        run_log.info("Built decoder from config: %s -> %s",
                     decoder_name,
                     type(decoder).__name__)
    else:
        run_log.info("Using provided decoder: %s",
                     type(decoder).__name__)

    if isinstance(optimizer, TrainingOptimizerConfig):
        optimizer_name = optimizer.name
        optimizer_cls = OPTIMIZERS.get(optimizer_name) # Use registry as optimizer is built as a factory
        optimizer_factory = build_optimizer_factory(optimizer)
        run_log.info("Built optimizer factory from config: %s -> %s",
                     optimizer_name,
                     optimizer_cls.__name__)
    else:
        optimizer_factory = optimizer
        run_log.info("Using provided optimizer factory: %s",
                     getattr(optimizer, "__name__", type(optimizer).__name__))

    # Compression warning
    variational_head_in_features = encoder.output_dim
    compression_ratio = variational_head_in_features / latent_dim

    if compression_ratio >= 8:
        run_log.warning(
            "Strong encoder-to-latent bottleneck detected: "
            "in_features=%d, latent_dim=%d, compression_ratio=%.1f. "
            "This is valid VAE behavior, but may strongly limit representation capacity.",
            variational_head_in_features,
            latent_dim,
            compression_ratio,
        )
    elif compression_ratio >= 4:
        run_log.info(
            "Meaningful encoder-to-latent compression detected: "
            "in_features=%d, latent_dim=%d, compression_ratio=%.1f.",
            variational_head_in_features,
            latent_dim,
            compression_ratio,
        )

    variational_head = GaussianVariationalHead(
        in_features=encoder.output_dim,
        latent_dim=latent_dim,
    )
    run_log.info("Built variational head: %s",
                 type(variational_head).__name__)

    losses_by_role = build_loss_terms(
        {
            "reconstruction": reconstruction_losses,
            "regularization": regularization_losses,
            "custom_objective": custom_objective_losses,
        },
    )

    return VAE(
        encoder=encoder,
        decoder=decoder,
        variational_head=variational_head,
        reconstruction_losses=losses_by_role["reconstruction"],
        regularization_losses=losses_by_role["regularization"],
        custom_objective_losses=losses_by_role["custom_objective"],
        optimizer_factory=optimizer_factory,
        prediction_reconstruction_latent_source=(
            prediction_reconstruction_latent_source
        ),
    )
