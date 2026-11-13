"""Contrastive representation losses."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class TripletMarginContrastiveLoss(nn.Module):
    """Triplet margin loss over anchor, positive, and negative vectors.

    Parameters
    ----------
    margin
        Positive margin between positive and negative pair distances.
    p
        Norm degree used for pairwise distances.
    eps
        Small constant used for numerical stability in distance computation.
    swap
        Whether to allow swapping the positive-negative distance as in PyTorch TripletMarginLoss.
    reduction
        Mean or sum across triplets.
    """

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()

        valid_reductions = ("mean", "sum")
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, "
                f"got {reduction!r}."
            )

        self.loss = nn.TripletMarginLoss(
            margin=margin,
            p=p,
            eps=eps,
            swap=swap,
            reduction=reduction,
        )

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar triplet-margin loss for anchor, positive, and negative embeddings.

        Supply matching tensors of shape (batch, features), or individual vectors.
        """
        return self.loss(anchor, positive, negative)