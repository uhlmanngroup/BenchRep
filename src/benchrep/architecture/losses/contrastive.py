"""Contrastive representation losses."""

from __future__ import annotations

from typing import Literal

import math

import torch
from torch import nn
from pytorch_metric_learning import losses as metric_losses


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


class NTXentContrastiveLoss(nn.Module):
    """Symmetric NT-Xent for two views of the same batch.

    Row i in each view represents the same sample. Both views serve as
    anchors, and the other samples in the minibatch supply the negatives.

    Parameters
    ----------
    temperature
        Positive divisor for cosine similarities. Lower values give more
        weight to the most similar negatives.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")

        self.loss = metric_losses.SelfSupervisedLoss(
            metric_losses.NTXentLoss(temperature=temperature),
            symmetric=True,
        )

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        """Compute the loss for two matching (batch, features) tensors."""
        if view_a.ndim != 2 or view_a.shape != view_b.shape:
            raise ValueError(
                "NT-Xent expects two tensors of the same (batch, features) shape."
            )
        if view_a.shape[0] < 2:
            raise ValueError(
                "NT-Xent needs at least two distinct samples per batch "
                "to supply negatives."
            )
        return self.loss(view_a, view_b)