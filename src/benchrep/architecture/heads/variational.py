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

from benchrep.architecture.heads.base import BaseHead


class GaussianVariationalHead(BaseHead):
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