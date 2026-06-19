from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from dataclasses import dataclass

import torch
from torch import nn

from benchrep.architecture.utils import (
    IntPair,
    validate_int_pair,
    resolve_activation,
)
from benchrep.architecture.encoders.base import BaseEncoder


@dataclass(frozen=True)
class Conv2DBlockSpec:
    """Resolved configuration for one Conv2D encoder block.

    This is an internal architecture contract. User-facing Conv2DEncoder params
    are normalized into block specs before PyTorch modules are constructed.
    """

    out_channels: int
    kernel_size: IntPair = 3
    stride: IntPair = 1
    padding: IntPair = 0
    dilation: IntPair = 1
    bias: bool = True
    normalization: str | None = None # 'batchnorm', 'instancenorm', 'groupnorm'
    normalization_groups: int | None = None
    activation: type[nn.Module] = nn.ReLU
    dropout: float = 0.0
    pooling: str | None = None # 'max' and 'avg'
    pooling_kernel_size: IntPair = 2
    pooling_stride: IntPair | None = None
    pooling_padding: IntPair = 0


class Conv2DBlock(nn.Module):
    """One Conv2D encoder block built from a resolved block spec.

    The block order is:

    Conv2d -> optional normalization -> activation -> optional dropout
    -> optional pooling
    """

    def __init__(
        self,
        in_channels: int,
        spec: Conv2DBlockSpec,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if spec.out_channels <= 0:
            raise ValueError(
                f"spec.out_channels must be positive, got {spec.out_channels}."
            )
        if not 0.0 <= spec.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {spec.dropout}.")

        validate_int_pair(spec.kernel_size, name="kernel_size", min_value=0, allow_equal_min=False)
        validate_int_pair(spec.stride, name="stride", min_value=0, allow_equal_min=False)
        validate_int_pair(spec.padding, name="padding", min_value=0)
        validate_int_pair(spec.dilation, name="dilation", min_value=0, allow_equal_min=False)

        normalization = (
            spec.normalization.lower()
            if spec.normalization is not None
            else None
        )
        valid_normalizations = (None, "batchnorm", "instancenorm", "groupnorm")

        if normalization not in valid_normalizations:
            raise ValueError(
                "normalization must be one of "
                f"{valid_normalizations}, got {spec.normalization!r}."
            )

        groupnorm_groups: int | None = None
        if normalization == "groupnorm":
            groupnorm_groups = spec.normalization_groups
            out_channels = spec.out_channels
            if groupnorm_groups is None:
                raise ValueError("normalization_groups is required when normalization='groupnorm'.")

            if groupnorm_groups <= 0:
                raise ValueError(
                    f"normalization_groups must be positive, got {groupnorm_groups}."
                )

            if out_channels % groupnorm_groups != 0:
                raise ValueError(
                    "For GroupNorm, out_channels must be divisible by normalization_groups. "
                    f"Got out_channels={out_channels}, normalization_groups={groupnorm_groups}."
                )
        elif spec.normalization_groups is not None:
            raise ValueError(
                "normalization_groups is only valid when normalization='groupnorm'."
            )

        pooling = spec.pooling.lower() if spec.pooling is not None else None
        valid_pooling = (None, "max", "avg")

        if pooling not in valid_pooling:
            raise ValueError(
                f"pooling must be one of {valid_pooling}, got {spec.pooling!r}."
            )
        validate_int_pair(
            spec.pooling_kernel_size,
            name="pooling_kernel_size",
            min_value=0,
            allow_equal_min=False)
        if spec.pooling_stride is not None:
            validate_int_pair(
                spec.pooling_stride,
                name="pooling_stride",
                min_value=0,
                allow_equal_min=False)
        validate_int_pair(spec.pooling_padding, name="pooling_padding", min_value=0)

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=spec.out_channels,
                kernel_size=spec.kernel_size,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
                bias=spec.bias,
            )
        ]

        if normalization == "batchnorm":
            layers.append(nn.BatchNorm2d(spec.out_channels))
        elif normalization == "instancenorm":
            layers.append(nn.InstanceNorm2d(spec.out_channels))
        elif normalization == "groupnorm":
            layers.append(
                nn.GroupNorm(
                    num_groups=groupnorm_groups,
                    num_channels=spec.out_channels,
                )
            )

        layers.append(spec.activation())

        if spec.dropout > 0:
            layers.append(nn.Dropout2d(spec.dropout))

        if pooling == "max":
            layers.append(
                nn.MaxPool2d(
                    kernel_size=spec.pooling_kernel_size,
                    stride=spec.pooling_stride,
                    padding=spec.pooling_padding,
                )
            )
        elif pooling == "avg":
            layers.append(
                nn.AvgPool2d(
                    kernel_size=spec.pooling_kernel_size,
                    stride=spec.pooling_stride,
                    padding=spec.pooling_padding,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Conv2DEncoder(BaseEncoder):
    """Conv2D encoder mapping image-like tensors to latent vectors.

    The encoder expects one input sample to have shape ``(channels, height, width)``
    and receives batched tensors as ``[batch, channels, height, width]``.

    Two configuration modes are supported:

    1. Channels mode:
       Pass ``channels=[...]``. Each integer creates one Conv2D block with that
       value as ``out_channels``. All blocks share the same convolution,
       normalization, activation, dropout, and pooling settings.

    2. Blocks mode:
       Pass ``blocks=[...]``. Each mapping creates one Conv2D block. Top-level
       block parameters act as shared defaults, and values provided inside an
       individual block override those defaults.

    Examples
    --------
    Channels mode::

        Conv2DEncoder(
            input_shape=(1, 64, 64),
            output_dim=128,
            channels=[32, 64, 128],
            kernel_size=3,
            stride=2,
            padding=1,
            normalization="batchnorm",
            activation="relu",
        )

    Blocks mode with only the final block customized::

        Conv2DEncoder(
            input_shape=(1, 64, 64),
            output_dim=128,
            kernel_size=3,
            stride=1,
            padding=1,
            activation="relu",
            normalization="batchnorm",
            blocks=[
                {"out_channels": 32},
                {"out_channels": 64},
                {"out_channels": 128, "stride": 2, "dropout": 0.1},
            ],
        )

    Internally, user-facing parameters are normalized into ``Conv2DBlockSpec``
    objects. The encoder then builds a convolutional stack, infers the final
    convolutional feature shape with a dummy input, flattens the features, and
    projects them to ``output_dim``.
    """
    def __init__(
            self,
            input_shape: tuple[int, int, int],
            output_dim: int,
            *,
            channels: Sequence[int] | None = None,
            blocks: Sequence[Mapping[str, Any]] | None = None,
            kernel_size: IntPair = 3,
            stride: IntPair = 1,
            padding: IntPair = 0,
            dilation: IntPair = 1,
            bias: bool = True,
            normalization: str | None = None,
            normalization_groups: int | None = None,
            activation: str | type[nn.Module] | None = None,
            dropout: float = 0.0,
            pooling: str | None = None,
            pooling_kernel_size: IntPair = 2,
            pooling_stride: IntPair | None = None,
            pooling_padding: IntPair = 0,
    ) -> None:
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(
                "Conv2DEncoder input_shape must be a 3-tuple "
                f"(channels, height, width), got {input_shape}."
            )
        if any(dim <= 0 for dim in input_shape):
            raise ValueError(
                f"All input_shape dimensions must be positive, got {input_shape}."
            )
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        # Constructor owns config interpretation
        block_specs = self._resolve_block_specs(
            channels=channels,
            blocks=blocks,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            normalization=normalization,
            normalization_groups=normalization_groups,
            activation=activation,
            dropout=dropout,
            pooling=pooling,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
        )

        self._input_shape = tuple(input_shape)
        self._output_dim = output_dim
        self.block_specs = block_specs

        layers: list[nn.Module] = []
        in_channels = self._input_shape[0]

        for spec in self.block_specs:
            layers.append(
                Conv2DBlock(
                    in_channels=in_channels,
                    spec=spec,
                )
            )
            in_channels = spec.out_channels

        self.conv_stack = nn.Sequential(*layers)

        self._feature_shape = self._infer_feature_shape()
        self.flatten_dim = int(torch.tensor(self._feature_shape).prod().item())
        self.projection = nn.Linear(self.flatten_dim, output_dim)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return self._feature_shape

    @staticmethod
    def _resolve_block_specs(
            *,
            channels: Sequence[int] | None = None,
            blocks: Sequence[Mapping[str, Any]] | None = None,
            kernel_size: IntPair = 3,
            stride: IntPair = 1,
            padding: IntPair = 0,
            dilation: IntPair = 1,
            bias: bool = True,
            normalization: str | None = None,
            normalization_groups: int | None = None,
            activation: str | type[nn.Module] | None = None,
            dropout: float = 0.0,
            pooling: str | None = None,
            pooling_kernel_size: IntPair = 2,
            pooling_stride: IntPair | None = None,
            pooling_padding: IntPair = 0,
    ) -> tuple[Conv2DBlockSpec, ...]:
        """Resolve user-facing Conv2D block configuration.

        ``channels`` and ``blocks`` are mutually exclusive.

        In channels mode, each value in ``channels`` becomes one block and all
        other block parameters are shared across blocks.

        In blocks mode, each mapping in ``blocks`` becomes one block. The shared
        keyword arguments passed to this method are used as defaults, and
        per-block mappings override only the fields they specify.
        """
        if channels is None and blocks is None:
            raise ValueError("Conv2DEncoder requires either 'channels' or 'blocks'.")

        if channels is not None and blocks is not None:
            raise ValueError(
                "Conv2DEncoder accepts either 'channels' or 'blocks', not both."
            )

        activation_cls = resolve_activation(activation)

        shared_defaults = {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "bias": bias,
            "normalization": normalization,
            "normalization_groups": normalization_groups,
            "activation": activation_cls,
            "dropout": dropout,
            "pooling": pooling,
            "pooling_kernel_size": pooling_kernel_size,
            "pooling_stride": pooling_stride,
            "pooling_padding": pooling_padding,
        }

        # Channel mode branch (homogeneous architecture)
        if channels is not None:
            if len(channels) == 0:
                raise ValueError("'channels' must contain at least one value.")

            block_specs = []
            for block_index, out_channels in enumerate(channels):
                if not isinstance(out_channels, int):
                    raise TypeError(
                        "'channels' values must be integers. "
                        f"Got {type(out_channels).__name__} at index {block_index}."
                    )

                block_specs.append(
                    Conv2DBlockSpec(
                        out_channels=out_channels,
                        **shared_defaults,
                    )
                )

            return tuple(block_specs)

        # Block mode branch (heterogeneous architecture)
        if len(blocks) == 0:
            raise ValueError("'blocks' must contain at least one block.")

        allowed_block_keys = {"out_channels"} | set(shared_defaults)
        block_specs = []

        for block_index, block in enumerate(blocks):
            unknown_keys = set(block) - allowed_block_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown Conv2D block keys at block index {block_index}: "
                    f"{sorted(unknown_keys)}."
                )

            if "out_channels" not in block:
                raise ValueError(
                    "Each Conv2D block must define 'out_channels'. "
                    f"Missing in block index {block_index}."
                )

            block_kwargs = dict(shared_defaults)
            block_kwargs.update(block)

            out_channels = block_kwargs["out_channels"]
            if not isinstance(out_channels, int):
                raise TypeError(
                    "Conv2D block 'out_channels' must be an integer. "
                    f"Got {type(out_channels).__name__} at block index {block_index}."
                )

            block_kwargs["activation"] = resolve_activation(block_kwargs["activation"])

            block_specs.append(
                Conv2DBlockSpec(**block_kwargs)
            )

        return tuple(block_specs)

    def _infer_feature_shape(self) -> tuple[int, int, int]:
        """Infer convolutional feature shape before flattening/projection."""
        dummy = torch.zeros(1, *self.input_shape)

        # Set model to eval mode, to guard against BatchNorm erroring if
        # dummy hits [1, C, 1, 1]
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                features = self.conv_stack(dummy)
        except RuntimeError as error:
            raise ValueError(
                "Conv2DEncoder could not run a dummy input through the conv stack. "
                "Check input_shape, kernel_size, stride, padding, dilation, and pooling."
            ) from error
        finally:
            self.train(was_training)

        if features.ndim != 4:
            raise ValueError(
                "Conv2DEncoder conv_stack must return a 4D tensor "
                f"[batch, channels, height, width], got shape {tuple(features.shape)}."
            )

        feature_shape = tuple(int(dim) for dim in features.shape[1:])

        if any(dim <= 0 for dim in feature_shape):
            raise ValueError(
                f"Inferred feature_shape must be positive, got {feature_shape}."
            )

        return feature_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Conv2DEncoder expects a 4D tensor [batch, channels, height, width], "
                f"got shape {tuple(x.shape)}."
            )

        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected input shape per sample {self.input_shape}, "
                f"got {tuple(x.shape[1:])}."
            )

        features = self.conv_stack(x)
        features = torch.flatten(features, start_dim=1)

        if features.shape[-1] != self.flatten_dim:
            raise ValueError(
                f"Expected flattened feature dimension {self.flatten_dim}, "
                f"got {features.shape[-1]}."
            )

        return self.projection(features)