from __future__ import annotations

import math

from collections.abc import Sequence

import torch
from torch import nn

from benchrep.architecture.utils import resolve_activation
from benchrep.architecture.encoders.base import BaseEncoder


class MLPEncoder(BaseEncoder):
    """MLP encoder mapping inputs to latent vectors.

    Parameters
    ----------
    input_shape:
        Shape of one input sample, excluding the batch dimension. Inputs with more
        than two dimensions are flattened across all non-batch dimensions, e.g.
        ``[B, C, H, W] -> [B, C*H*W]``.
    output_dim:
        Dimensionality of the final latent representation.
    hidden_dims:
        Sizes of the hidden fully connected layers.
    activation:
        Activation used after each hidden linear layer. Can be a supported string,
        an ``nn.Module`` class, or ``None`` to use the default ReLU.
    dropout:
        Dropout probability applied after activation. Set to 0.0 to disable.
    normalization:
        Optional normalization after each hidden linear layer. Supported values are:
        None, "batchnorm", and "layernorm".
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        activation: str | type[nn.Module] | None = None,
        dropout: float = 0.0,
        normalization: str | None = None,
    ) -> None:
        super().__init__()

        if len(input_shape) == 0:
            raise ValueError("input_shape must contain at least one dimension.")
        if any(dim <= 0 for dim in input_shape):
            raise ValueError(
                f"All input_shape dimensions must be positive, got {input_shape}."
            )
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"All hidden_dims must be positive, got {hidden_dims}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        activation_cls = resolve_activation(activation)

        if normalization is not None:
            if not isinstance(normalization, str):
                raise TypeError(
                    "normalization must be None or a string, "
                    f"got {type(normalization).__name__}."
                )
            normalization = normalization.lower()
        valid_normalizations = (None, "batchnorm", "layernorm")

        if normalization not in valid_normalizations:
            raise ValueError(
                f"normalization must be one of {valid_normalizations}, got {normalization!r}."
            )

        self._input_shape = tuple(input_shape)
        self.input_dim = math.prod(self._input_shape)
        self._output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation_cls

        layers: list[nn.Module] = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(activation_cls())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim} from "
                f"input_shape={self.input_shape}, got {x.shape[-1]}."
            )

        return self.net(x)