from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseHead(nn.Module, ABC):
    """Base interface for model heads."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Transform component input into one or more head outputs."""
        raise NotImplementedError