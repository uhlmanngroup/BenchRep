from __future__ import annotations

import math

from collections.abc import Sequence

import torch
from torch import nn

from benchrep.architecture.decoders.base import BaseDecoder


class MLPDecoder(BaseDecoder):
    """MLP decoder mapping latent vectors to reconstructed outputs.

    Parameters
    ----------
    input_dim:
        Dimensionality of the latent input representation.
    output_shape:
        Shape of one reconstructed output sample, excluding the batch dimension.
        Decoder outputs are reshaped from flat vectors, e.g.
        ``[B, C*H*W] -> [B, C, H, W]``.
    hidden_dims:
        Sizes of the hidden fully connected layers.
    activation:
        Activation module class used after each hidden linear layer.
    dropout:
        Dropout probability applied after activation. Set to 0.0 to disable.
    normalization:
        Optional normalization after each hidden linear layer. Supported values are:
        None, "batchnorm", and "layernorm".
    output_activation:
        Optional activation applied after the final linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_shape: tuple[int, ...],
        hidden_dims: Sequence[int] = (256, 512),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        normalization: str | None = None,
        output_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if len(output_shape) == 0:
            raise ValueError("output_shape must contain at least one dimension.")
        if any(dim <= 0 for dim in output_shape):
            raise ValueError(
                f"All output_shape dimensions must be positive, got {output_shape}."
            )
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"All hidden_dims must be positive, got {hidden_dims}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        if normalization is not None:
            normalization = normalization.lower()
        valid_normalizations = (None, "batchnorm", "layernorm")

        if normalization not in valid_normalizations:
            raise ValueError(
                f"normalization must be one of {valid_normalizations}, got {normalization!r}."
            )

        self._input_dim = input_dim
        self._output_shape = tuple(output_shape)
        self.output_dim = math.prod(self._output_shape)
        self.hidden_dims = tuple(hidden_dims)
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation
        self.output_activation = output_activation

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))

        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected decoder input feature dimension {self.input_dim}, "
                f"got {z.shape[-1]}."
            )

        output = self.net(z)
        return output.reshape(output.shape[0], *self.output_shape)