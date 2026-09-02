from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from benchrep.architecture.heads.base import BaseHead
from benchrep.architecture.utils import (
    resolve_activation,
    resolve_normalization,
    validate_normalization,
)


class MLPHead(BaseHead):
    """MLP head mapping feature vectors to output vectors.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input feature vector.
    output_dim:
        Dimensionality of the output vector.
    hidden_dims:
        Sizes of the hidden fully connected layers. An empty sequence produces
        a single linear mapping from ``input_dim`` to ``output_dim``.
    activation:
        Activation used after each hidden linear layer. Can be a supported string,
        an ``nn.Module`` class, or ``None`` to use the default ReLU.
    dropout:
        Dropout probability applied after activation. Set to 0.0 to disable.
    normalization:
        Optional normalization after each hidden linear layer. Supported values are:
        None, "batchnorm", "layernorm", and "rmsnorm".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (),
        activation: str | type[nn.Module] | None = None,
        dropout: float = 0.0,
        normalization: str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(input_dim, int) or isinstance(input_dim, bool):
            raise TypeError(
                f"input_dim must be an integer, got {type(input_dim).__name__}."
            )
        if not isinstance(output_dim, int) or isinstance(output_dim, bool):
            raise TypeError(
                f"output_dim must be an integer, got {type(output_dim).__name__}."
            )

        for index, hidden_dim in enumerate(hidden_dims):
            if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool):
                raise TypeError(
                    "hidden_dims values must be integers. "
                    f"Got {type(hidden_dim).__name__} at index {index}."
                )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"All hidden_dims must be positive, got {hidden_dims}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if len(hidden_dims) == 0:
            if activation is not None:
                raise ValueError(
                    "activation requires at least one hidden layer."
                )
            if normalization is not None:
                raise ValueError(
                    "normalization requires at least one hidden layer."
                )
            if dropout > 0:
                raise ValueError(
                    "dropout requires at least one hidden layer."
                )

        activation_cls = resolve_activation(activation)

        normalization = validate_normalization(
            normalization,
            layout="vector",
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation_cls

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            normalization_layer = resolve_normalization(
                normalization,
                num_features=hidden_dim,
                layout="vector",
            )

            if normalization_layer is not None:
                layers.append(normalization_layer)

            layers.append(activation_cls())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} expects a torch.Tensor, "
                f"got {type(x).__name__}."
            )
        if x.ndim != 2:
            raise ValueError(
                "MLPHead expects a 2D tensor [batch, features], "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, "
                f"got {x.shape[-1]}."
            )

        return self.net(x)