"""Supervised regression losses."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class MSERegressionLoss(nn.Module):
    """Mean squared error for continuous predictions."""

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()

        valid_reductions = ("mean", "sum")
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, "
                f"got {reduction!r}."
            )

        self.loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have the same shape, got "
                f"prediction.shape={tuple(prediction.shape)} and "
                f"target.shape={tuple(target.shape)}."
            )

        return self.loss(prediction, target)