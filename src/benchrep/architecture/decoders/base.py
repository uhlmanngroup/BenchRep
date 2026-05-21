from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDecoder(nn.Module, ABC):
    """Base interface for decoders.

    Decoders map latent representations back to data space.
    """

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Expected dimensionality of the latent input."""
        raise NotImplementedError

    @property
    def output_shape(self) -> tuple[int, ...] | None:
        """Shape of reconstructed outputs, excluding batch dimension, if defined."""
        return None

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation into reconstructed output."""
        raise NotImplementedError