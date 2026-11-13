"""Variational latent head for VAE-style models.

This module converts deterministic encoder features into the parameters of a
diagonal Gaussian latent distribution and samples from it using the
reparameterization trick.

Expected use:

    encoder_features -> VariationalHead -> z_sample / z_mu / z_logvar

The posterior mean (`z_mu`) is usually the deterministic embedding used for
downstream evaluation, while `z_sample` is the sampled latent vector used
during VAE training.
"""

from __future__ import annotations

import torch
from torch import nn


class GaussianVariationalHead(nn.Module):
    """Map encoder features to diagonal Gaussian posterior parameters and a sample.

    Two learned linear maps produce the posterior mean and log-variance.
    Reparameterization draws a sample in both training and evaluation mode.


    Parameters
    ----------
    in_features
        Number of encoder features per sample.
    latent_dim
        Number of latent dimensions; all 3 returned tensors have this width.
    """

    def __init__(self, in_features: int, latent_dim: int) -> None:
        super().__init__()

        if not isinstance(in_features, int) or isinstance(in_features, bool):
            raise TypeError(
                f"in_features must be an integer, got {type(in_features).__name__}."
            )
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool):
            raise TypeError(
                f"latent_dim must be an integer, got {type(latent_dim).__name__}."
            )
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")

        self.in_features = in_features
        self.latent_dim = latent_dim

        self.mu_layer = nn.Linear(in_features, latent_dim)
        self.logvar_layer = nn.Linear(in_features, latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return z_mu, z_logvar, and z_sample for a batch of encoder features.

        Parameters
        ----------
        x
            Tensor with shape (batch, in_features).

        Returns
        -------
        dict[str, torch.Tensor]
            Posterior means, log-variances, and reparameterized samples, each
            with shape (batch, latent_dim).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} expects a torch.Tensor, "
                f"got {type(x).__name__}."
            )

        if x.ndim != 2:
            raise ValueError(
                "GaussianVariationalHead expects a 2D tensor [batch, features], "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input feature dimension {self.in_features}, "
                f"got {x.shape[-1]}."
            )

        z_mu = self.mu_layer(x)
        z_logvar = self.logvar_layer(x)
        z_sample = self.reparameterize(z_mu=z_mu, z_logvar=z_logvar)

        return {
            "z_sample": z_sample,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
        }

    @staticmethod
    def reparameterize(
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample z from N(z_mu, diag(exp(z_logvar))) using reparameterization."""
        if not isinstance(z_mu, torch.Tensor):
            raise TypeError(
                f"z_mu must be a torch.Tensor, got {type(z_mu).__name__}."
            )
        if not isinstance(z_logvar, torch.Tensor):
            raise TypeError(
                f"z_logvar must be a torch.Tensor, got {type(z_logvar).__name__}."
            )
        if z_mu.shape != z_logvar.shape:
            raise ValueError(
                "z_mu and z_logvar must have identical shapes, "
                f"got {tuple(z_mu.shape)} and {tuple(z_logvar.shape)}."
            )

        z_sigma = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_sigma)

        return z_mu + eps * z_sigma