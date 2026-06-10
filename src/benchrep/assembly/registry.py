from __future__ import annotations

from typing import Any

import torch
from torchvision import transforms

from lightning.pytorch.loggers import (
    CSVLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
)

from benchrep.architecture.data import MNISTDataset
from benchrep.architecture.decoders import MLPDecoder
from benchrep.architecture.encoders import MLPEncoder
from benchrep.architecture.losses import (
    MSEReconstructionLoss,
    MAEReconstructionLoss,
    GaussianKLDivergenceLoss,
)
from benchrep.architecture.models import (
    Autoencoder,
    VAE,
)
from benchrep.evaluation.clustering import run_kmeans, run_leiden
from benchrep.evaluation.reductions import run_pca, run_tsne, run_umap


class Registry:
    """Name-to-object registry used by builders.

    The registry maps string names from config files to Python classes or
    callables that can be instantiated by builders.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}
        self._canonical_keys: dict[str, str] = {}

    def register(self, key: str, item: Any, *aliases: str) -> None:
        canonical_key = self._normalize_key(key)
        # Silently collapse duplicates
        names = tuple(
            dict.fromkeys(self._normalize_key(name) for name in (key, *aliases))
        )

        # Refuse overwrites
        for name in names:
            if name in self._items:
                raise KeyError(
                    f"{self.name} registry already contains key {name!r}. "
                    "Choose a different name or remove the existing registration."
                )

        for name in names:
            self._items[name] = item
            self._canonical_keys[name] = canonical_key

    def get(self, key: str) -> Any:
        # Retrieve a registered object by name, with a debuggable error for unknown keys.
        key = self._normalize_key(key)

        if key not in self._items:
            available = tuple(sorted(self._items))
            raise KeyError(
                f"Unknown {self.name} key {key!r}. "
                f"Available options: {available}."
            )

        return self._items[key]

    def resolve_key(self, key: str) -> str:
        """Resolve a registered key or alias to its canonical registry key."""

        key = self._normalize_key(key)

        if key not in self._canonical_keys:
            available = tuple(sorted(self._items))
            raise KeyError(
                f"Unknown {self.name} key {key!r}. "
                f"Available options: {available}."
            )

        return self._canonical_keys[key]

    def create(self, key: str, **kwargs: Any) -> Any:
        # Retrieve a registered class/callable and instantiate it with keyword arguments.
        item = self.get(key)
        return item(**kwargs)

    def keys(self) -> tuple[str, ...]:
        # Return registered keys in deterministic order for errors, debugging, and validation.
        return tuple(sorted(self._items))

    def canonical_keys(self) -> tuple[str, ...]:
        # Return canonical registered keys in deterministic order.
        return tuple(sorted(set(self._canonical_keys.values())))

    @staticmethod
    def _normalize_key(key: str) -> str:
        if not isinstance(key, str):
            raise TypeError(f"Registry keys must be strings, got {type(key).__name__}.")

        key = key.lower().strip().replace("-", "_")

        if not key:
            raise ValueError("Registry key must be a non-empty string.")

        return key


# -------------------------
# INSTANTIATION
# -------------------------
# Data
DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
# Architecture and training
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
MODELS = Registry("model")
RECONSTRUCTION_LOSSES = Registry("reconstruction loss")
REGULARIZATION_LOSSES = Registry("regularization loss")
OPTIMIZERS = Registry("optimizer")
LOGGERS = Registry("logger")
# Evaluation
EVAL_REDUCTIONS = Registry("reduction")
EVAL_CLUSTERING_METHODS = Registry("clustering method")
EVAL_INTERNAL_CLUSTERING_METRICS = Registry("internal clustering metrics")
EVAL_EXTERNAL_CLUSTERING_METRICS = Registry("external clustering metrics")
EVAL_EMBEDDING_METRICS = Registry("embedding metrics")
EVAL_RECONSTRUCTION_METRICS = Registry("reconstruction metrics")


# -------------------------
# REGISTRATION
# -------------------------

# --- Data ---
DATASETS.register("mnist", MNISTDataset)

TRANSFORMS.register("to_tensor", transforms.ToTensor)

# --- Architecture and training ---
ENCODERS.register("mlp", MLPEncoder)

DECODERS.register("mlp", MLPDecoder)

MODELS.register("autoencoder", Autoencoder, "ae")
MODELS.register(
    "vae",
    VAE,
    "variational_autoencoder",
    "variational_ae",
    "gaussian_vae",
)

RECONSTRUCTION_LOSSES.register("mse", MSEReconstructionLoss, "l2")
RECONSTRUCTION_LOSSES.register("mae", MAEReconstructionLoss, "l1")

REGULARIZATION_LOSSES.register(
    "gaussian_kl",
    GaussianKLDivergenceLoss,
    "kl",
    "kld",
    "kldiv",
    "kl_div",
    "gaussian_kld",
    "gaussian_kldiv",
    "gaussian_kl_div",
)

OPTIMIZERS.register("adam", torch.optim.Adam)
OPTIMIZERS.register("adamw", torch.optim.AdamW)
OPTIMIZERS.register("sgd", torch.optim.SGD)

LOGGERS.register("csv", CSVLogger, "csvlogger")
LOGGERS.register("wandb", WandbLogger, "wandblogger")
LOGGERS.register(
    "tensorboard",
    TensorBoardLogger,
    "tensorboardlogger",
    "tb",
    "tblogger",
)
LOGGERS.register("mlflow", MLFlowLogger, "mlflowlogger")

# --- Evaluation ---
# Reductions
EVAL_REDUCTIONS.register(
    "pca",
    run_pca,
    "principal_component_analysis",
    "principal_components",
)
EVAL_REDUCTIONS.register("umap", run_umap)
EVAL_REDUCTIONS.register("tsne", run_tsne, "t_sne")
# Clustering
EVAL_CLUSTERING_METHODS.register("kmeans", run_kmeans, "k_means")
EVAL_CLUSTERING_METHODS.register("leiden", run_leiden)
# Internal clustering metrics
EVAL_INTERNAL_CLUSTERING_METRICS.register(
    "silhouette",
    silhouette_score,
    "silhouette_score",
)
EVAL_INTERNAL_CLUSTERING_METRICS.register(
    "calinski_harabasz",
    calinski_harabasz_score,
    "calinski_harabasz_score",
    "ch",
    "ch_score",
)
EVAL_INTERNAL_CLUSTERING_METRICS.register(
    "davies_bouldin",
    davies_bouldin_score,
    "davies_bouldin_score",
    "db",
    "db_score",
)
# External clustering metrics
EVAL_EXTERNAL_CLUSTERING_METRICS.register(
    "adjusted_mutual_info",
    adjusted_mutual_info_score,
    "adjusted_mutual_info_score",
    "adjusted_mutual_information",
    "adjusted_mutual_information_score",
    "adj_mutual_info",
    "adj_mutual_info_score",
    "ami",
    "ami_score",
)
EVAL_EXTERNAL_CLUSTERING_METRICS.register(
    "adjusted_rand_index",
    adjusted_rand_score,
    "adjusted_rand_score",
    "adjusted_rand",
    "adj_rand_index",
    "adj_rand_score",
    "ari",
    "ari_score",
)
EVAL_EXTERNAL_CLUSTERING_METRICS.register(
    "homogeneity",
    homogeneity_score,
    "homogeneity_score",
)