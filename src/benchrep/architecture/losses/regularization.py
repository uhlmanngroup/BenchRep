"""Regularization losses.

This module contains loss terms that regularize model structure or latent
representations.
"""

from __future__ import annotations

import torch
from torch import nn


class GaussianKLDivergenceLoss(nn.Module):
    """KL divergence from a diagonal Gaussian posterior to a standard normal prior.

    This is the standard regularization term used in vanilla VAEs:

        KL[N(z_mu, diag(exp(z_logvar))) || N(0, I)]

    Args:
        reduction:
            How to reduce the per-sample KL values. Supported values are
            ``"mean"``, ``"sum"``, and ``"none"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be one of {'mean', 'sum', 'none'}, "
                f"got {reduction!r}."
            )

        self.reduction = reduction

    def forward(
        self,
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Return KL divergence for Gaussian posterior parameters.

        Args:
            z_mu:
                Posterior mean tensor with shape ``(..., latent_dim)``.
            z_logvar:
                Posterior log-variance tensor with shape
                ``(..., latent_dim)``.

        Returns:
            Reduced KL divergence value, or per-sample KL values if
            ``reduction="none"``.
        """

        if z_mu.shape != z_logvar.shape:
            raise ValueError(
                "z_mu and z_logvar must have the same shape, "
                f"got {tuple(z_mu.shape)} and {tuple(z_logvar.shape)}."
            )

        kl_per_sample = -0.5 * torch.sum(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp(),
            dim=-1,
        )

        if self.reduction == "mean":
            return kl_per_sample.mean()

        if self.reduction == "sum":
            return kl_per_sample.sum()

        return kl_per_sample