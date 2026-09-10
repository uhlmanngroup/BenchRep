from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("benchrep")
except PackageNotFoundError:
    __version__ = "0+unknown"

from benchrep.workflows.train import train_ae, train_vae, train_composite
from benchrep.workflows.predict import predict_ae, predict_vae
from benchrep.workflows.evaluate import evaluate
from benchrep.discovery.registry import (
    list_registries,
    list_registered_components,
    inspect_registry,
)
from benchrep.discovery.config import inspect_config


__all__ = [
    "__version__",
    "train_ae",
    "train_vae",
    "train_composite",
    "predict_ae",
    "predict_vae",
    "evaluate",
    "list_registries",
    "list_registered_components",
    "inspect_registry",
    "inspect_config",
]