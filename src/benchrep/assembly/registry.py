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

    def register(self, key: str, item: Any, *aliases: str) -> None:
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
TRANSFORMS = Registry("transform")
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
MODELS = Registry("model")
RECONSTRUCTION_LOSSES = Registry("reconstruction loss")
OPTIMIZERS = Registry("optimizer")
LOGGERS = Registry("logger")

# Register currently supported components
DATASETS.register("mnist", MNISTDataset)

TRANSFORMS.register("to_tensor", transforms.ToTensor)

ENCODERS.register("mlp", MLPEncoder)

DECODERS.register("mlp", MLPDecoder)

MODELS.register("autoencoder", Autoencoder)

RECONSTRUCTION_LOSSES.register("mse", MSEReconstructionLoss, "l2")

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