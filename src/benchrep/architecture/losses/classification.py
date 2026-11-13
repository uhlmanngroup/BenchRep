"""Supervised classification losses."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class CrossEntropyClassificationLoss(nn.Module):
    """Cross-entropy loss for categorical prediction logits.

    Parameters
    ----------
    ignore_index
        Target class index excluded from the loss when targets are class indices.
    label_smoothing
        Amount of label smoothing passed to torch.nn.CrossEntropyLoss.
    reduction
        Mean or sum reduction.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()

        valid_reductions = ("mean", "sum")
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, "
                f"got {reduction!r}."
            )

        self.loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar cross-entropy from unnormalized logits and targets.

        Inputs follow torch.nn.CrossEntropyLoss: prediction contains class logits;
        target contains integer class indices or class probabilities of matching shape.
        The class dimension is 1 for batched inputs.
        """
        return self.loss(prediction, target)