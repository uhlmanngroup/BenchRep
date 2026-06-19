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
from benchrep.architecture.decoders.base import BaseDecoder


@dataclass(frozen=True)
class UpsampleConv2DBlockSpec:
    """Resolved configuration for one Upsample + Conv2D decoder block.

    This is an internal architecture contract. User-facing UpsampleConv2DDecoder params
    are normalized into block specs before PyTorch modules are constructed.
    """

    out_channels: int

    # Upsampling
    upsample_scale_factor: IntPair = 2
    upsample_mode: str = "nearest" # 'nearest' and 'bilinear'
    upsample_align_corners: bool | None = None

    # Conv2D
    kernel_size: IntPair = 3
    stride: IntPair = 1
    padding: IntPair = 1
    dilation: IntPair = 1
    bias: bool = True

    # Post-conv layers
    normalization: str | None = None  # 'batchnorm', 'instancenorm', 'groupnorm'
    normalization_groups: int | None = None
    activation: type[nn.Module] = nn.ReLU
    dropout: float = 0.0


class UpsampleConv2DBlock(nn.Module):
    """One Upsample + Conv2D decoder block built from a resolved block spec.

    The block order is:

    Upsample -> Conv2d -> optional normalization -> activation -> optional dropout
    """

    def __init__(
            self,
            in_channels: int,
            spec: UpsampleConv2DBlockSpec,
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

        validate_int_pair(spec.upsample_scale_factor, name="upsample_scale_factor", min_value=1)
        validate_int_pair(spec.kernel_size, name="kernel_size", min_value=0, allow_equal_min=False)
        validate_int_pair(spec.stride, name="stride", min_value=0, allow_equal_min=False)
        validate_int_pair(spec.padding, name="padding", min_value=0)
        validate_int_pair(spec.dilation, name="dilation", min_value=0, allow_equal_min=False)

        upsample_mode = spec.upsample_mode.lower()
        valid_upsampling_modes = ("nearest", "bilinear")

        if upsample_mode not in valid_upsampling_modes:
            raise ValueError(
                "upsample_mode must be one of "
                f"{valid_upsampling_modes}, got {spec.upsample_mode!r}."
            )
        if upsample_mode == "nearest" and spec.upsample_align_corners is not None:
            raise ValueError(
                "spec.upsample_align_corners incompatible with spec.upsample_mode `nearest`."
            )

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

        layers: list[nn.Module] = [
            nn.Upsample(
                scale_factor=spec.upsample_scale_factor,
                mode=upsample_mode,
                align_corners=spec.upsample_align_corners,
            ),
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

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpsampleConv2DDecoder(BaseDecoder):
    """Upsample-convolutional decoder mapping latent vectors to image-like tensors.

    The decoder expects latent inputs with shape ``[batch, input_dim]`` and
    reconstructs outputs with shape ``[batch, channels, height, width]``.

    Internally, the latent vector is first projected with a linear layer and reshaped
    to ``initial_shape``. The resulting feature map is then passed through an
    upsampling convolutional stack and finally projected to ``output_shape[0]``
    channels with a final Conv2D layer.

    ``initial_shape`` is the decoder-side starting feature-map shape
    ``(channels, height, width)``. In normal config-driven model construction, this
    should be inferred by the model builder from ``encoder.feature_shape`` rather
    than provided manually in the decoder config.

    Two configuration modes are supported:

    1. Channels mode:
       Pass ``channels=[...]``. Each integer creates one UpsampleConv2D block with
       that value as ``out_channels``. All blocks share the same upsampling,
       convolution, normalization, activation, and dropout settings.

    2. Blocks mode:
       Pass ``blocks=[...]``. Each mapping creates one UpsampleConv2D block.
       Top-level block parameters act as shared defaults, and values provided inside
       an individual block override those defaults.

    Examples
    --------
    Channels mode::

        UpsampleConv2DDecoder(
            input_dim=32,
            output_shape=(1, 28, 28),
            initial_shape=(64, 7, 7),
            channels=[32, 16],
            upsample_scale_factor=2,
            kernel_size=3,
            padding=1,
            normalization="batchnorm",
            activation="relu",
            output_activation="sigmoid",
        )

    Blocks mode with only the final block customized::

        UpsampleConv2DDecoder(
            input_dim=32,
            output_shape=(1, 28, 28),
            initial_shape=(64, 7, 7),
            upsample_scale_factor=2,
            kernel_size=3,
            padding=1,
            activation="relu",
            normalization="batchnorm",
            blocks=[
                {"out_channels": 32},
                {"out_channels": 16, "dropout": 0.1},
            ],
            output_activation="sigmoid",
        )

    Internally, user-facing parameters are normalized into
    ``UpsampleConv2DBlockSpec`` objects. The decoder then builds a projection layer,
    reshapes projected latent vectors to ``initial_shape``, applies the upsampling
    convolutional stack, and validates the inferred reconstruction shape against
    ``output_shape`` with a dummy latent input.
    """
    def __init__(
        self,
        input_dim: int,
        output_shape: tuple[int, int, int],
        *,
        initial_shape: tuple[int, int, int],
        channels: Sequence[int] | None = None,
        blocks: Sequence[Mapping[str, Any]] | None = None,
        upsample_scale_factor: IntPair = 2,
        upsample_mode: str = "nearest",
        upsample_align_corners: bool | None = None,
        kernel_size: IntPair = 3,
        stride: IntPair = 1,
        padding: IntPair = 1,
        dilation: IntPair = 1,
        bias: bool = True,
        normalization: str | None = None,
        normalization_groups: int | None = None,
        activation: str | type[nn.Module] | None = None,
        dropout: float = 0.0,
        output_kernel_size: IntPair = 1,
        output_conv_padding: IntPair = 0,
        output_bias: bool = True,
        output_activation: str | type[nn.Module] | None = None,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")

        if len(output_shape) != 3:
            raise ValueError(
                "UpsampleConv2DDecoder output_shape must be a 3-tuple "
                f"(channels, height, width), got {output_shape}."
            )
        if any(dim <= 0 for dim in output_shape):
            raise ValueError(
                f"All output_shape dimensions must be positive, got {output_shape}."
            )

        if len(initial_shape) != 3:
            raise ValueError(
                "initial_shape must be a 3-tuple "
                f"(channels, height, width), got {initial_shape}."
            )
        if any(dim <= 0 for dim in initial_shape):
            raise ValueError(
                f"All initial_shape dimensions must be positive, got {initial_shape}."
            )

        validate_int_pair(
            output_kernel_size,
            name="output_kernel_size",
            min_value=0,
            allow_equal_min=False,
        )
        validate_int_pair(
            output_conv_padding,
            name="output_conv_padding",
            min_value=0,
        )

        block_specs = self._resolve_block_specs(
            channels=channels,
            blocks=blocks,
            upsample_scale_factor=upsample_scale_factor,
            upsample_mode=upsample_mode,
            upsample_align_corners=upsample_align_corners,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            normalization=normalization,
            normalization_groups=normalization_groups,
            activation=activation,
            dropout=dropout,
        )

        output_activation_cls = (
            resolve_activation(output_activation)
            if output_activation is not None
            else None
        )

        self._input_dim = input_dim
        self._output_shape = tuple(output_shape)
        self._initial_shape = tuple(initial_shape)
        self.block_specs = block_specs
        self.output_activation = output_activation_cls

        self.initial_flatten_dim = int(torch.tensor(self._initial_shape).prod().item())
        self.projection = nn.Linear(input_dim, self.initial_flatten_dim)

        layers: list[nn.Module] = []
        in_channels = self._initial_shape[0]

        for spec in self.block_specs:
            layers.append(
                UpsampleConv2DBlock(
                    in_channels=in_channels,
                    spec=spec,
                )
            )
            in_channels = spec.out_channels

        self.conv_stack = nn.Sequential(*layers)

        output_channels = self._output_shape[0]

        output_layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=output_kernel_size,
                stride=1,
                padding=output_conv_padding,
                bias=output_bias,
            )
        ]

        if output_activation_cls is not None:
            output_layers.append(output_activation_cls())

        self.output_layer = nn.Sequential(*output_layers)

        inferred_output_shape = self._infer_output_shape()
        if inferred_output_shape != self.output_shape:
            raise ValueError(
                "UpsampleConv2DDecoder inferred output shape does not match "
                f"requested output_shape. Got {inferred_output_shape}, "
                f"expected {self.output_shape}. Check initial_shape, number of "
                "blocks, upsample_scale_factor, kernel_size, padding, and "
                "output_kernel_size/output_conv_padding."
            )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self._output_shape

    @property
    def initial_shape(self) -> tuple[int, int, int]:
        return self._initial_shape

    @staticmethod
    def _resolve_block_specs(
        *,
        channels: Sequence[int] | None = None,
        blocks: Sequence[Mapping[str, Any]] | None = None,
        upsample_scale_factor: IntPair = 2,
        upsample_mode: str = "nearest",
        upsample_align_corners: bool | None = None,
        kernel_size: IntPair = 3,
        stride: IntPair = 1,
        padding: IntPair = 1,
        dilation: IntPair = 1,
        bias: bool = True,
        normalization: str | None = None,
        normalization_groups: int | None = None,
        activation: str | type[nn.Module] | None = None,
        dropout: float = 0.0,
    ) -> tuple[UpsampleConv2DBlockSpec, ...]:
        """Resolve user-facing UpsampleConv2D block configuration.

        ``channels`` and ``blocks`` are mutually exclusive.

        In channels mode, each value in ``channels`` becomes one block and all
        other block parameters are shared across blocks.

        In blocks mode, each mapping in ``blocks`` becomes one block. The shared
        keyword arguments passed to this method are used as defaults, and
        per-block mappings override only the fields they specify.
        """
        if channels is None and blocks is None:
            raise ValueError(
                "UpsampleConv2DDecoder requires either 'channels' or 'blocks'."
            )

        if channels is not None and blocks is not None:
            raise ValueError(
                "UpsampleConv2DDecoder accepts either 'channels' or 'blocks', "
                "not both."
            )

        activation_cls = resolve_activation(activation)

        shared_defaults = {
            "upsample_scale_factor": upsample_scale_factor,
            "upsample_mode": upsample_mode,
            "upsample_align_corners": upsample_align_corners,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "bias": bias,
            "normalization": normalization,
            "normalization_groups": normalization_groups,
            "activation": activation_cls,
            "dropout": dropout,
        }

        # Channel mode branch: homogeneous architecture.
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
                    UpsampleConv2DBlockSpec(
                        out_channels=out_channels,
                        **shared_defaults,
                    )
                )

            return tuple(block_specs)

        # Block mode branch: heterogeneous architecture.
        if len(blocks) == 0:
            raise ValueError("'blocks' must contain at least one block.")

        allowed_block_keys = {"out_channels"} | set(shared_defaults)
        block_specs = []

        for block_index, block in enumerate(blocks):
            unknown_keys = set(block) - allowed_block_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown UpsampleConv2D block keys at block index {block_index}: "
                    f"{sorted(unknown_keys)}."
                )

            if "out_channels" not in block:
                raise ValueError(
                    "Each UpsampleConv2D block must define 'out_channels'. "
                    f"Missing in block index {block_index}."
                )

            block_kwargs = dict(shared_defaults)
            block_kwargs.update(block)

            out_channels = block_kwargs["out_channels"]
            if not isinstance(out_channels, int):
                raise TypeError(
                    "UpsampleConv2D block 'out_channels' must be an integer. "
                    f"Got {type(out_channels).__name__} at block index {block_index}."
                )

            block_kwargs["activation"] = resolve_activation(block_kwargs["activation"])

            block_specs.append(
                UpsampleConv2DBlockSpec(**block_kwargs)
            )

        return tuple(block_specs)

    def _decode_unchecked(self, z: torch.Tensor) -> torch.Tensor:
        """Decode without public input/output validation."""
        output = self.projection(z)
        output = output.reshape(output.shape[0], *self.initial_shape)
        output = self.conv_stack(output)
        return self.output_layer(output)

    def _infer_output_shape(self) -> tuple[int, int, int]:
        """Infer reconstructed output shape from a dummy latent input."""
        dummy = torch.zeros(1, self.input_dim)

        # Set model to eval mode, to guard against BatchNorm erroring if
        # dummy hits [1, C, 1, 1]
        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                output = self._decode_unchecked(dummy)
        except RuntimeError as error:
            raise ValueError(
                "UpsampleConv2DDecoder could not run a dummy latent vector "
                "through the decoder. Check input_dim, initial_shape, "
                "upsample settings, convolution settings, and output_shape."
            ) from error
        finally:
            self.train(was_training)

        if output.ndim != 4:
            raise ValueError(
                "UpsampleConv2DDecoder must return a 4D tensor "
                f"[batch, channels, height, width], got shape {tuple(output.shape)}."
            )

        output_shape = tuple(int(dim) for dim in output.shape[1:])

        if any(dim <= 0 for dim in output_shape):
            raise ValueError(
                f"Inferred output_shape must be positive, got {output_shape}."
            )

        return output_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected decoder input feature dimension {self.input_dim}, "
                f"got {z.shape[-1]}."
            )

        output = self._decode_unchecked(z)

        if tuple(output.shape[1:]) != self.output_shape:
            raise ValueError(
                f"Expected decoder output shape per sample {self.output_shape}, "
                f"got {tuple(output.shape[1:])}."
            )

        return output