from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from math import prod
from typing import Any

import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.nn import functional as F

from benchrep.architecture.decoders import BaseDecoder
from benchrep.architecture.encoders import BaseEncoder
from benchrep.architecture.losses import BaseCustomObjectiveLoss
from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentPort,
    ComponentTensorResult,
)
from benchrep.architecture.losses.composite_model_contracts import (
    LossComponent,
    LossTensorPort,
)
from benchrep.assembly.registries.core import (
    DATASETS,
    TRANSFORMS,
    ENCODERS,
    DECODERS,
    RECONSTRUCTION_LOSSES,
    REGULARIZATION_LOSSES,
    CUSTOM_OBJECTIVE_LOSSES,
    OPTIMIZERS,
    LOGGERS,
    CALLBACKS,
    EVAL_INTERNAL_CLUSTERING_METRICS,
    EVAL_EXTERNAL_CLUSTERING_METRICS,
    EVAL_EMBEDDING_METRICS,
    EVAL_RECONSTRUCTION_METRICS,
)
from benchrep.evaluation.metrics import EvaluationMetric
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


class CustomCombinedObjectiveLoss(BaseCustomObjectiveLoss):
    def __init__(
        self,
        regularization_weight: float = 0.0001,
    ) -> None:
        super().__init__()
        self.regularization_weight = regularization_weight

    def forward(
        self,
        *,
        batch: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        COMPONENT_CALLS["custom_objective_loss"] += 1

        target = batch["x"]

        if not isinstance(target, torch.Tensor):
            raise TypeError("batch['x'] must be a torch.Tensor.")

        reconstruction_loss = F.mse_loss(
            model_output["reconstruction"],
            target,
        )

        regularization_loss = -0.5 * torch.mean(
            1
            + model_output["z_logvar"]
            - model_output["z_mu"].pow(2)
            - model_output["z_logvar"].exp()
        )

        return (
            reconstruction_loss
            + self.regularization_weight * regularization_loss
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


class CustomRegisteredCallback(Callback):
    def __init__(self, marker: str) -> None:
        COMPONENT_CALLS["callback_init"] += 1
        self.marker = marker

    def on_train_start(
        self,
        trainer: Any,
        pl_module: Any,
    ) -> None:
        COMPONENT_CALLS["callback_train_start"] += 1


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
        ArchitectureComponent(
            component=CustomRegisteredEncoder,
            runtime_inputs=(
                ComponentPort(
                    name="x",
                    supported_structures=("image",),
                ),
            ),
            runtime_result=ComponentTensorResult(
                supported_structures=("vector",),
            ),
        ),
    )
    DECODERS.register(
        "custom_test_decoder",
        ArchitectureComponent(
            component=CustomRegisteredDecoder,
            runtime_inputs=(
                ComponentPort(
                    name="z",
                    supported_structures=("vector",),
                ),
            ),
            runtime_result=ComponentTensorResult(
                supported_structures=("image",),
            ),
        ),
    )
    RECONSTRUCTION_LOSSES.register(
        "custom_test_reconstruction_loss",
        LossComponent(
            component=CustomReconstructionLoss,
            runtime_inputs=(
                LossTensorPort(
                    name="reconstruction",
                    supported_roles=("reconstruction_image",),
                ),
                LossTensorPort(
                    name="target",
                    supported_roles=(
                        "sample_image",
                        "positive_image",
                        "negative_image",
                    ),
                ),
            ),
        ),
    )
    REGULARIZATION_LOSSES.register(
        "custom_test_regularization_loss",
        LossComponent(
            component=CustomRegularizationLoss,
            runtime_inputs=(
                LossTensorPort(
                    name="z_mu",
                    supported_roles=(
                        "embedding_vector",
                        "continuous_auxiliary_vector",
                    ),
                ),
                LossTensorPort(
                    name="z_logvar",
                    supported_roles=("continuous_auxiliary_vector",),
                ),
            ),
        ),
    )
    CUSTOM_OBJECTIVE_LOSSES.register(
        "custom_test_objective_loss",
        LossComponent(CustomCombinedObjectiveLoss),
    )
    OPTIMIZERS.register(
        "custom_test_optimizer",
        create_custom_optimizer,
    )
    LOGGERS.register(
        "custom_test_logger",
        CustomLogger,
    )
    CALLBACKS.register(
        "custom_test_callback",
        CustomRegisteredCallback,
    )
    EVAL_INTERNAL_CLUSTERING_METRICS.register(
        "custom_test_internal_metric",
        EvaluationMetric(
            fn=custom_internal_clustering_metric,
            result_kind="scalar",
        ),
    )
    EVAL_EXTERNAL_CLUSTERING_METRICS.register(
        "custom_test_external_metric",
        EvaluationMetric(
            fn=custom_external_clustering_metric,
            result_kind="scalar",
        ),
    )
    EVAL_EMBEDDING_METRICS.register(
        "custom_test_embedding_metric",
        EvaluationMetric(
            fn=custom_embedding_metric,
            result_kind="vector",
            vector_axis="embedding_dimension",
        ),
    )
    EVAL_RECONSTRUCTION_METRICS.register(
        "custom_test_reconstruction_metric",
        EvaluationMetric(
            fn=custom_reconstruction_metric,
            result_kind="scalar",
        ),
    )