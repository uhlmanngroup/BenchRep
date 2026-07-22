from benchrep.workflows.train import train_ae, train_vae
from benchrep.workflows.predict import predict_ae, predict_vae
from benchrep.workflows.evaluate import evaluate
from benchrep.assembly.registries.discovery import (
    list_registries,
    list_registered_components,
)


__all__ = [
    "train_ae",
    "train_vae",
    "predict_ae",
    "predict_vae",
    "evaluate",
    "list_registries",
    "list_registered_components",
]