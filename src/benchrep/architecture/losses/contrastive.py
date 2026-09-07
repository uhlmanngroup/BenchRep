"""Contrastive representation losses."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class TripletMarginContrastiveLoss(nn.Module):
    """Triplet margin loss over anchor, positive, and negative vectors."""

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
        return self.loss(anchor, positive, negative)