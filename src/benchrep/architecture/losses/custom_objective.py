from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class BaseCustomObjectiveLoss(nn.Module, ABC):
    """Base interface for unrestricted custom objective losses.

    Custom objective losses receive the complete batch and model-output
    mappings, allowing them to combine inputs, annotations, reconstructions,
    embeddings, latent variables, or other model-family-specific outputs.

    Implementations must return a scalar loss tensor.
    """

    @abstractmethod
    def forward(
        self,
        *,
        batch: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the custom objective loss."""
        raise NotImplementedError