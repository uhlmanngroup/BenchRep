"""Regularization losses.

This module contains loss terms that regularize model structure or latent
representations.
"""

from __future__ import annotations

import torch
from torch import nn


class GaussianKLDivergenceLoss(nn.Module):
    """KL divergence from a diagonal Gaussian posterior to a standard normal prior.

    Computes -0.5 * sum(1 + z_logvar - z_mu**2 - exp(z_logvar)) over the
    last (latent) dimension, then reduces the remaining dimensions.


    Parameters
    ----------
    reduction
        Mean or sum of the per-sample KL values.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        valid_reductions = ("mean", "sum")
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, got {reduction!r}."
            )

        self.reduction = reduction

    def forward(
        self,
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Return the reduced KL divergence for matching posterior parameter tensors.

        Parameters
        ----------
        z_mu
            Posterior means with shape (..., latent_dim).
        z_logvar
            Posterior log-variances with the same shape as z_mu.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence, summed over latent dimensions and reduced
            over samples using the configured mean or sum.
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

        return kl_per_sample.sum()