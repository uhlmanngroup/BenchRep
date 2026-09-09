from __future__ import annotations

import inspect
from typing import Any

from benchrep.architecture.composite_model_component_contracts import ArchitectureComponent
from benchrep.architecture.decoders import BaseDecoder

from benchrep.architecture.encoders import BaseEncoder
from benchrep.architecture.heads import BaseHead
from benchrep.assembly.registries import ENCODERS, DECODERS
from benchrep.assembly.registries.core import HEADS
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.schemas import TrainingEncoderConfig, TrainingDecoderConfig


def build_head(
    name: str,
    *,
    params: dict[str, Any] | None = None,
) -> BaseHead:
    """Build a registered model head."""

    head_name = normalize_name(
        name,
        field_name="head.name",
    )

    head = HEADS.create(
        head_name,
        **(params or {}),
    )

    if not isinstance(head, BaseHead):
        raise TypeError(
            f"Registered head {head_name!r} produced "
            f"{type(head).__name__}, expected a BaseHead instance."
        )

    return head


def build_encoder(encoder_config: TrainingEncoderConfig) -> BaseEncoder:
    """Build a registered encoder from configuration.

    Parameters
    ----------
    encoder_config:
        Configuration selecting the registered encoder and its constructor
        parameters.

    Returns
    -------
    BaseEncoder
        Instantiated encoder.

    Raises
    ------
    TypeError
        If the registered encoder does not produce a ``BaseEncoder`` instance.
    """
    encoder_name = normalize_name(
        encoder_config.name,
        field_name="config.encoder.name",
    )

    encoder = ENCODERS.create(
        encoder_name,
        **encoder_config.params,
    )

    if not isinstance(encoder, BaseEncoder):
        raise TypeError(
            f"Registered encoder {encoder_name!r} produced "
            f"{type(encoder).__name__}, expected a BaseEncoder instance."
        )

    return encoder


def build_decoder(
    decoder_config: TrainingDecoderConfig,
    input_dim: int | None = None,
    encoder: BaseEncoder | None = None,
) -> BaseDecoder:
    """Build a registered decoder from configuration.

    Decoder constructor parameters are taken from ``decoder_config.params``.
    Architecture-dependent parameters may additionally be supplied or inferred
    by the builder.

    If the selected decoder accepts ``input_dim``, an explicitly configured
    value takes precedence, provided it agrees with ``input_dim`` when both are
    supplied.

    If the selected decoder accepts ``initial_shape``, an explicitly configured
    value is used when available. Otherwise, BenchRep attempts to infer it from
    ``encoder.feature_shape``. If both are available, they must agree.

    Parameters
    ----------
    decoder_config:
        Configuration selecting the registered decoder and its constructor
        parameters.
    input_dim:
        Optional input dimensionality supplied by the surrounding model
        builder.
    encoder:
        Optional encoder instance used to infer ``initial_shape`` when supported
        by the selected decoder.

    Returns
    -------
    BaseDecoder
        Instantiated decoder.

    Raises
    ------
    ValueError
        If configured and supplied ``input_dim`` values disagree, if configured
        ``initial_shape`` disagrees with ``encoder.feature_shape``, or if a required
        ``input_dim`` or ``initial_shape`` cannot be resolved.
    """
    decoder_name = normalize_name(
        decoder_config.name,
        field_name="config.decoder.name",
    )
    decoder_entry = DECODERS.get(decoder_name)
    assert isinstance(decoder_entry, ArchitectureComponent)

    decoder_cls = decoder_entry.component
    decoder_params = dict(decoder_config.params)
    decoder_signature = inspect.signature(decoder_cls)

    # Wire decoder input dimensionality from the supplied input dimension.
    input_dim_parameter = decoder_signature.parameters.get("input_dim")

    if input_dim_parameter is not None:
        configured_input_dim = decoder_params.get("input_dim")

        if configured_input_dim is not None and input_dim is not None:
            if configured_input_dim != input_dim:
                raise ValueError(
                    f"Decoder {decoder_name!r} builder received input_dim="
                    f"{configured_input_dim}, but supplied input_dim={input_dim}."
                )

        elif configured_input_dim is None and input_dim is not None:
            decoder_params["input_dim"] = input_dim

        elif (
                configured_input_dim is None
                and input_dim_parameter.default is inspect.Parameter.empty
        ):
            raise ValueError(
                f"Decoder {decoder_name!r} requires 'input_dim', but none was "
                "provided in config or supplied to the builder."
            )

    initial_shape_parameter = decoder_signature.parameters.get("initial_shape")

    if initial_shape_parameter is not None:
        configured_initial_shape = decoder_params.get("initial_shape")
        if configured_initial_shape is not None:
            if not isinstance(configured_initial_shape, (list, tuple)):
                raise TypeError(
                    f"Decoder {decoder_name!r} initial_shape must be a list or tuple."
                )
            configured_initial_shape = tuple(configured_initial_shape)

        inferred_initial_shape = (
            encoder.feature_shape
            if encoder is not None
            else None
        )

        if configured_initial_shape is not None and inferred_initial_shape is not None:
            if configured_initial_shape != inferred_initial_shape:
                raise ValueError(
                    f"Decoder {decoder_name!r} builder received initial_shape="
                    f"{configured_initial_shape}, but provided encoder.feature_shape="
                    f"{inferred_initial_shape}."
                )

        elif configured_initial_shape is None and inferred_initial_shape is not None:
            decoder_params["initial_shape"] = inferred_initial_shape

        elif (
            configured_initial_shape is None
            and initial_shape_parameter.default is inspect.Parameter.empty
        ):
            raise ValueError(
                f"Decoder {decoder_name!r} requires 'initial_shape', but none was "
                "provided and it could not be inferred from encoder.feature_shape."
            )

    return DECODERS.create(
        decoder_name,
        **decoder_params,
    )
