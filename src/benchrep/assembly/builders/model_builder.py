from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import inspect

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
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.schemas import (
    TrainingConfig,
    DecoderConfig,
    EncoderConfig,
    LossTermConfig,
    OptimizerConfig,
)
from benchrep.assembly.registries.core import (
    DECODERS,
    ENCODERS,
    MODELS,
    OPTIMIZERS,
    RECONSTRUCTION_LOSSES,
    REGULARIZATION_LOSSES,
    CUSTOM_OBJECTIVE_LOSSES,
)


def build_model(
    config: TrainingConfig,
    *,
    prediction_reconstruction_latent_source: (
        Literal["mean", "sample"] | None
    ) = None,
) -> L.LightningModule:
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
        if prediction_reconstruction_latent_source is not None:
            raise ValueError(
                "`prediction_reconstruction_latent_source` is only supported "
                "when building a VAE."
            )

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
        if config.decoder is None:
            raise ValueError("VAE requires a decoder config section.")
        if prediction_reconstruction_latent_source is None:
            prediction_reconstruction_latent_source: Literal["mean", "sample"] = "mean"

        model = build_vae(
            encoder=config.encoder,
            decoder=config.decoder,
            optimizer=config.optimizer,
            latent_dim=config.model.params["latent_dim"],
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
    encoder: EncoderConfig | BaseEncoder,
    decoder: DecoderConfig | BaseDecoder,
    optimizer: (
            OptimizerConfig |
            Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
    ),
    reconstruction_losses: dict[str, LossTermConfig | LossTerm],
    custom_objective_losses: dict[
        str,
        LossTermConfig | LossTerm,
    ],
) -> Autoencoder:
    run_log = get_run_logger()

    # Resolve configs objs into instantiated components where needed
    if isinstance(encoder, EncoderConfig):
        encoder_name = encoder.name
        encoder = _build_encoder(encoder)
        run_log.info("Built encoder from config: %s -> %s",
                     encoder_name,
                     type(encoder).__name__)
    else:
        run_log.info("Using provided encoder: %s",
                     type(encoder).__name__)

    if isinstance(decoder, DecoderConfig):
        decoder_name = decoder.name
        decoder = _build_decoder(
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

    if isinstance(optimizer, OptimizerConfig):
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

    sources_by_role = {
        "reconstruction": {
            loss_name: (
                "provided"
                if isinstance(loss_spec, LossTerm)
                else "config"
            )
            for loss_name, loss_spec
            in reconstruction_losses.items()
        },
        "custom_objective": {
            loss_name: (
                "provided"
                if isinstance(loss_spec, LossTerm)
                else "config"
            )
            for loss_name, loss_spec
            in custom_objective_losses.items()
        },
    }

    reconstruction_losses = _build_reconstruction_losses(
        reconstruction_losses
    )
    custom_objective_losses = _build_custom_objective_losses(
        custom_objective_losses
    )

    _log_resolved_losses(
        losses_by_role={
            "reconstruction": reconstruction_losses,
            "custom_objective": custom_objective_losses,
        },
        sources_by_role=sources_by_role,
    )

    return Autoencoder(
        encoder=encoder,
        decoder=decoder,
        reconstruction_losses=reconstruction_losses,
        custom_objective_losses=custom_objective_losses,
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
    custom_objective_losses: dict[
        str,
        LossTermConfig | LossTerm,
    ],
    prediction_reconstruction_latent_source: (
            Literal["mean", "sample"]
    ) = "mean",
) -> VAE:
    run_log = get_run_logger()

    # Resolve configs objs into instantiated components where needed
    if isinstance(encoder, EncoderConfig):
        encoder_name = encoder.name
        encoder = _build_encoder(encoder)
        run_log.info("Built encoder from config: %s -> %s",
                     encoder_name,
                     type(encoder).__name__)
    else:
        run_log.info("Using provided encoder: %s",
                     type(encoder).__name__)

    if isinstance(decoder, DecoderConfig):
        decoder_name = decoder.name
        decoder = _build_decoder(
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

    if isinstance(optimizer, OptimizerConfig):
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

    sources_by_role = {
        "reconstruction": {
            loss_name: (
                "provided"
                if isinstance(loss_spec, LossTerm)
                else "config"
            )
            for loss_name, loss_spec
            in reconstruction_losses.items()
        },
        "regularization": {
            loss_name: (
                "provided"
                if isinstance(loss_spec, LossTerm)
                else "config"
            )
            for loss_name, loss_spec
            in regularization_losses.items()
        },
        "custom_objective": {
            loss_name: (
                "provided"
                if isinstance(loss_spec, LossTerm)
                else "config"
            )
            for loss_name, loss_spec
            in custom_objective_losses.items()
        },
    }

    reconstruction_losses = _build_reconstruction_losses(
        reconstruction_losses
    )
    regularization_losses = _build_regularization_losses(
        regularization_losses
    )
    custom_objective_losses = _build_custom_objective_losses(
        custom_objective_losses
    )

    _log_resolved_losses(
        losses_by_role={
            "reconstruction": reconstruction_losses,
            "regularization": regularization_losses,
            "custom_objective": custom_objective_losses,
        },
        sources_by_role=sources_by_role,
    )

    return VAE(
        encoder=encoder,
        decoder=decoder,
        variational_head=variational_head,
        reconstruction_losses=reconstruction_losses,
        regularization_losses=regularization_losses,
        custom_objective_losses=custom_objective_losses,
        optimizer_factory=optimizer_factory,
        prediction_reconstruction_latent_source=(
            prediction_reconstruction_latent_source
        ),
    )


def _build_encoder(encoder_config: EncoderConfig) -> BaseEncoder:
    encoder_name = normalize_name(
        encoder_config.name,
        field_name="config.encoder.name",
    )

    return ENCODERS.create(encoder_name, **encoder_config.params)


def _build_decoder(
    decoder_config: DecoderConfig,
    input_dim: int,
    encoder: BaseEncoder | None = None,
) -> BaseDecoder:
    """Build a decoder from config and wire model-dependent dimensions.

    The decoder config supplies user-facing decoder parameters, while this helper
    injects dimensions that are determined by the surrounding model assembly.

    ``input_dim`` is always added to the decoder parameters. For ordinary
    decoders, such as MLP decoders, this is sufficient.

    Spatial decoders also require an ``initial_shape`` constructor argument:
    the feature-map shape used to reshape the projected latent vector before
    convolutional decoding. This shape should not be user-configured. If the
    decoder constructor declares ``initial_shape``, this helper infers it from
    ``encoder.feature_shape`` and passes it during construction.

    Raises
    ------
    ValueError
        If ``initial_shape`` is provided manually in decoder config, or if a
        decoder requires ``initial_shape`` but it cannot be inferred from the
        encoder.
    """
    decoder_name = normalize_name(
        decoder_config.name,
        field_name="config.decoder.name",
    )
    decoder_cls = DECODERS.get(decoder_name)
    decoder_params = dict(decoder_config.params)

    # Wire decoder input dimensionality from the supplied input dimension.
    decoder_params["input_dim"] = input_dim

    # Infer and pass initial_shape from encoder.feature_shape if needed.
    decoder_signature = inspect.signature(decoder_cls)
    if "initial_shape" in decoder_signature.parameters:
        if "initial_shape" in decoder_params:
            raise ValueError(
                "'initial_shape' should not be provided in decoder config. "
                "It is inferred from encoder.feature_shape."
            )

        feature_shape = getattr(encoder, "feature_shape", None)

        if feature_shape is None:
            raise ValueError(
                f"Decoder {decoder_name!r} requires 'initial_shape', but it could "
                "not be inferred because the encoder has no 'feature_shape' attribute."
            )

        decoder_params["initial_shape"] = feature_shape

    return decoder_cls(**decoder_params)


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


def _build_custom_objective_losses(
    custom_objective_losses: dict[
        str,
        LossTermConfig | LossTerm,
    ],
) -> dict[str, LossTerm]:
    loss_terms: dict[str, LossTerm] = {}

    for loss_name, loss_spec in custom_objective_losses.items():
        if isinstance(loss_spec, LossTerm):
            loss_terms[loss_name] = loss_spec
            continue

        loss_terms[loss_name] = LossTerm(
            loss=CUSTOM_OBJECTIVE_LOSSES.create(
                loss_name,
                **loss_spec.params,
            ),
            weight=loss_spec.weight,
        )

    return loss_terms


def _log_resolved_losses(
    *,
    losses_by_role: dict[str, dict[str, LossTerm]],
    sources_by_role: dict[str, dict[str, str]],
) -> None:
    run_log = get_run_logger()

    descriptions = []

    for role, loss_terms in losses_by_role.items():
        if not loss_terms:
            continue

        sources = sources_by_role[role]

        terms = ", ".join(
            (
                f"{loss_name} ({sources[loss_name]})"
                f" -> {type(loss_term.loss).__name__}"
                f" (weight={loss_term.weight})"
            )
            for loss_name, loss_term in loss_terms.items()
        )

        descriptions.append(f"{role}=[{terms}]")

    run_log.info(
        "Resolved losses: %s",
        "; ".join(descriptions),
    )
