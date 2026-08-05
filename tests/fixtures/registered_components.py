from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from math import prod
from typing import Any

import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from torch import nn
from torch.nn import functional as F

from benchrep.architecture.decoders import BaseDecoder
from benchrep.architecture.encoders import BaseEncoder
from benchrep.assembly.registries.core import (
    DATASETS,
    TRANSFORMS,
    ENCODERS,
    DECODERS,
    RECONSTRUCTION_LOSSES,
    REGULARIZATION_LOSSES,
    OPTIMIZERS,
    LOGGERS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_EMBEDDING_METRICS,
    EVAL_RECONSTRUCTION_METRICS,
)
from tests.fixtures.datasets import TinySyntheticDataset


COMPONENT_CALLS: Counter[str] = Counter()


def reset_component_calls() -> None:
    COMPONENT_CALLS.clear()


class CustomRegisteredDataset(TinySyntheticDataset):
    def __init__(self, **params: Any) -> None:
        COMPONENT_CALLS["dataset_init"] += 1
        super().__init__(**params)

    def __getitem__(self, index: int) -> dict[str, Any]:
        COMPONENT_CALLS["dataset_getitem"] += 1
        return super().__getitem__(index)


def create_custom_transform(
    scale: float = 1.0,
):
    COMPONENT_CALLS["transform_factory"] += 1

    def transform(x: torch.Tensor) -> torch.Tensor:
        COMPONENT_CALLS["transform_call"] += 1
        return x * scale

    return transform


class CustomRegisteredEncoder(BaseEncoder):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_dim: int,
    ) -> None:
        super().__init__()
        COMPONENT_CALLS["encoder_init"] += 1

        self._input_shape = tuple(input_shape)
        self._output_dim = output_dim
        self.projection = nn.Linear(prod(self._input_shape), output_dim)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        COMPONENT_CALLS["encoder_forward"] += 1
        return self.projection(torch.flatten(x, start_dim=1))


class CustomRegisteredDecoder(BaseDecoder):
    def __init__(
        self,
        input_dim: int,
        output_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        COMPONENT_CALLS["decoder_init"] += 1

        self._input_dim = input_dim
        self._output_shape = tuple(output_shape)
        self.projection = nn.Linear(input_dim, prod(self._output_shape))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        COMPONENT_CALLS["decoder_forward"] += 1
        reconstruction = torch.sigmoid(self.projection(z))

        return reconstruction.reshape(
            z.shape[0],
            *self._output_shape,
        )


class CustomReconstructionLoss(nn.Module):
    def forward(
        self,
        *,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        COMPONENT_CALLS["reconstruction_loss"] += 1
        return F.mse_loss(reconstruction, target)


class CustomRegularizationLoss(nn.Module):
    def forward(
        self,
        *,
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        COMPONENT_CALLS["regularization_loss"] += 1

        return -0.5 * torch.mean(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp()
        )


def create_custom_optimizer(
    parameters: Iterable[nn.Parameter],
    lr: float = 0.001,
) -> torch.optim.Optimizer:
    COMPONENT_CALLS["optimizer_factory"] += 1

    return torch.optim.SGD(
        parameters,
        lr=lr,
    )


class CustomLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        COMPONENT_CALLS["logger_init"] += 1

    @property
    def name(self) -> str:
        return "custom_test_logger"

    @property
    def version(self) -> str:
        return "0"

    def log_hyperparams(self, params: Any) -> None:
        COMPONENT_CALLS["logger_hyperparams"] += 1

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
    ) -> None:
        COMPONENT_CALLS["logger_metrics"] += 1

    def save(self) -> None:
        COMPONENT_CALLS["logger_save"] += 1

    def finalize(self, status: str) -> None:
        COMPONENT_CALLS["logger_finalize"] += 1


def custom_internal_clustering_metric(
    embeddings: Any,
    cluster_labels: Any,
    *,
    offset: float = 0.0,
) -> float:
    COMPONENT_CALLS["internal_clustering_metric"] += 1
    embedding_array = np.asarray(embeddings)

    return float(np.mean(np.var(embedding_array, axis=0)) + offset)


def custom_external_clustering_metric(
    reference_labels: Any,
    cluster_labels: Any,
    *,
    offset: float = 0.0,
) -> float:
    COMPONENT_CALLS["external_clustering_metric"] += 1

    return float(
        np.mean(
            np.asarray(reference_labels)
            == np.asarray(cluster_labels)
        )
        + offset
    )


def custom_embedding_metric(
    embeddings: Any,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    COMPONENT_CALLS["embedding_metric"] += 1
    embedding_array = np.asarray(embeddings)

    return np.mean(embedding_array, axis=0) * scale


def custom_reconstruction_metric(
    inputs: Any,
    reconstructions: Any,
    *,
    offset: float = 0.0,
) -> float:
    COMPONENT_CALLS["reconstruction_metric"] += 1

    return float(
        np.mean(
            np.abs(
                np.asarray(inputs) - np.asarray(reconstructions)
            )
        )
        + offset
    )


def register_custom_test_components() -> None:
    DATASETS.register(
        "custom_test_dataset",
        CustomRegisteredDataset,
    )
    TRANSFORMS.register(
        "custom_test_transform",
        create_custom_transform,
    )
    ENCODERS.register(
        "custom_test_encoder",
        CustomRegisteredEncoder,
    )
    DECODERS.register(
        "custom_test_decoder",
        CustomRegisteredDecoder,
    )
    RECONSTRUCTION_LOSSES.register(
        "custom_test_reconstruction_loss",
        CustomReconstructionLoss,
    )
    REGULARIZATION_LOSSES.register(
        "custom_test_regularization_loss",
        CustomRegularizationLoss,
    )
    OPTIMIZERS.register(
        "custom_test_optimizer",
        create_custom_optimizer,
    )
    LOGGERS.register(
        "custom_test_logger",
        CustomLogger,
    )
    EVAL_INTERNAL_CLUSTERING_METRICS.register(
        "custom_test_internal_metric",
        custom_internal_clustering_metric,
    )
    EVAL_EXTERNAL_CLUSTERING_METRICS.register(
        "custom_test_external_metric",
        custom_external_clustering_metric,
    )
    EVAL_EMBEDDING_METRICS.register(
        "custom_test_embedding_metric",
        custom_embedding_metric,
    )
    EVAL_RECONSTRUCTION_METRICS.register(
        "custom_test_reconstruction_metric",
        custom_reconstruction_metric,
    )