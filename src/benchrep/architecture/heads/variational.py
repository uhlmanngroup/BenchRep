"""Variational latent head for VAE-style models.

This module converts deterministic encoder features into the parameters of a
diagonal Gaussian latent distribution and samples from it using the
reparameterization trick.

Expected use:

    encoder_features -> VariationalHead -> z / z_mu / z_logvar

The posterior mean (`z_mu`) is usually the deterministic embedding used for
downstream evaluation, while `z` is the sampled latent vector used during VAE
training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GaussianVariationalHeadOutput:
    """Output container for a variational latent head.

    Attributes:
        z:
            Sampled latent vector produced by the reparameterization trick.
        z_mu:
            Mean of the approximate posterior distribution.
        z_logvar:
            Log-variance of the approximate posterior distribution.
    """

    z: torch.Tensor
    z_mu: torch.Tensor
    z_logvar: torch.Tensor


class GaussianVariationalHead(nn.Module):
    """Map encoder features to a sampled latent vector and posterior parameters.

    The head predicts the mean and log-variance of a diagonal Gaussian
    approximate posterior:

        q(z | x) = N(z_mu, diag(exp(z_logvar)))

    Sampling uses the reparameterization trick so gradients can flow through the
    stochastic latent variable during training (standard VAE architecture).

    Args:
        in_features:
            Number of input features from the encoder. The head expects a single
            feature vector per sample, with shape ``(batch_size, in_features)``.
            Multimodal or multi-encoder models should fuse features before this
            head, or use separate heads when modeling separate posteriors.
        latent_dim:
            Size of the latent embedding.
    """

    def __init__(self, in_features: int, latent_dim: int) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}.")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")

        self.in_features = in_features
        self.latent_dim = latent_dim

        self.mu_layer = nn.Linear(in_features, latent_dim)
        self.logvar_layer = nn.Linear(in_features, latent_dim)

    def forward(self, x: torch.Tensor) -> GaussianVariationalHeadOutput:
        """Return sampled latent vector and posterior parameters."""

        z_mu = self.mu_layer(x)
        z_logvar = self.logvar_layer(x)
        z = self.reparameterize(z_mu=z_mu, z_logvar=z_logvar)

        return GaussianVariationalHeadOutput(
            z=z,
            z_mu=z_mu,
            z_logvar=z_logvar,
        )

    @staticmethod
    def reparameterize(
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample z from N(z_mu, diag(exp(z_logvar))) using reparameterization."""

        z_sigma = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_sigma)

        return z_mu + eps * z_sigma