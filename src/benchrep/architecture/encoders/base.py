from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """Base interface for encoders.

    Encoders map input data to a latent representation.
    """

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        """Expected shape of one input sample, excluding batch dimension, if defined."""
        return None

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionality of the latent representation."""
        raise NotImplementedError

    @property
    def feature_shape(self) -> tuple[int, ...] | None:
        """Shape of intermediate features before projection/flattening, if defined."""
        return None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into latent representation."""
        raise NotImplementedError