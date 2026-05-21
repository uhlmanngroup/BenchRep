from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseReconstructionLoss(nn.Module, ABC):
    """Base interface for reconstruction losses.

    Reconstruction losses compare model reconstructions against the original
    input data.
    """

    @abstractmethod
    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        raise NotImplementedError


class MSEReconstructionLoss(BaseReconstructionLoss):
    """Mean squared error reconstruction loss.

    This is the standard L2 reconstruction loss used for continuous-valued
    reconstruction targets.

    Parameters
    ----------
    reduction:
        Reduction applied to the elementwise squared error. Supported values are
        "mean", "sum", and "none", matching ``torch.nn.MSELoss``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        valid_reductions = ("mean", "sum", "none")
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, got {reduction!r}."
            )

        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"reconstruction and target must have the same shape, got "
                f"reconstruction.shape={tuple(reconstruction.shape)} and "
                f"target.shape={tuple(target.shape)}."
            )

        return self.loss(reconstruction, target)
