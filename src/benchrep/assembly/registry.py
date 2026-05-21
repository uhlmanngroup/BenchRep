from __future__ import annotations

from typing import Any

import torch

from benchrep.architecture.data import MNISTDataset
from benchrep.architecture.decoders import MLPDecoder
from benchrep.architecture.encoders import MLPEncoder
from benchrep.architecture.losses import MSEReconstructionLoss
from benchrep.architecture.models import Autoencoder


class Registry:
    """Name-to-object registry used by builders.

    The registry maps string names from config files to Python classes or
    callables that can be instantiated by builders.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str, item: Any) -> None:
        # Add a new name-to-object mapping, refusing accidental overwrites.
        key = self._normalize_key(key)

        if key in self._items:
            raise KeyError(
                f"{self.name} registry already contains key {key!r}. "
                "Choose a different name or remove the existing registration."
            )

        self._items[key] = item

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

    def create(self, key: str, **kwargs: Any) -> Any:
        # Retrieve a registered class/callable and instantiate it with keyword arguments.
        item = self.get(key)
        return item(**kwargs)

    def keys(self) -> tuple[str, ...]:
        # Return registered keys in deterministic order for errors, debugging, and validation.
        return tuple(sorted(self._items))

    @staticmethod
    def _normalize_key(key: str) -> str:
        # Standardize registry keys so config names are case- and whitespace-tolerant.
        if not isinstance(key, str):
            raise TypeError(f"Registry keys must be strings, got {type(key).__name__}.")

        key = key.lower().strip()

        if not key:
            raise ValueError("Registry key must be a non-empty string.")

        return key


# Instantiate the registries
DATASETS = Registry("dataset")
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
MODELS = Registry("model")
RECONSTRUCTION_LOSSES = Registry("reconstruction loss")
OPTIMIZERS = Registry("optimizer")

# Register currently supported components
DATASETS.register("mnist", MNISTDataset)

ENCODERS.register("mlp", MLPEncoder)

DECODERS.register("mlp", MLPDecoder)

MODELS.register("autoencoder", Autoencoder)

RECONSTRUCTION_LOSSES.register("mse", MSEReconstructionLoss)
RECONSTRUCTION_LOSSES.register("l2", MSEReconstructionLoss)

OPTIMIZERS.register("adam", torch.optim.Adam)
OPTIMIZERS.register("adamw", torch.optim.AdamW)
OPTIMIZERS.register("sgd", torch.optim.SGD)