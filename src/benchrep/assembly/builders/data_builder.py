from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

import torch

from benchrep.records import get_run_logger
from benchrep.architecture.data import (
    BaseDataset,
    BenchRepDataModule,
    TransformPipeline,
    TransformStep,
)
from benchrep.assembly.schemas import (
    DataModuleConfig,
    TransformConfig,
    SupportedDatasetConfig,
)
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.registries.core import DATASETS, TRANSFORMS


@dataclass(frozen=True)
class TransformPipelineBundle:
    training: TransformPipeline
    preprocessing: TransformPipeline


def build_datamodule(
    *,
    dataset: BaseDataset,
    datamodule_config: DataModuleConfig,
    seed: int | None = None,
    stage: Literal["training", "prediction"],
    training_pipeline: TransformPipeline | None = None,
    preprocessing_pipeline: TransformPipeline | None = None,
) -> BenchRepDataModule:
    """Build a BenchRep datamodule around an instantiated dataset.

    For training, the dataset is assigned as the training dataset and may be
    divided into training and validation subsets according to ``val_fraction``.
    For prediction, the dataset is assigned directly as the prediction dataset.
    Workflow-specific configuration adjustments must be resolved before calling
    this builder.

    Parameters
    ----------
    dataset:
        Instantiated dataset to expose through the datamodule.
    datamodule_config:
        Batching, loading, and optional validation-splitting configuration.
    seed:
        Optional seed used for reproducible train-validation splitting.
    stage:
        Workflow stage determining whether the dataset is assigned for training
        or prediction.

    Returns
    -------
    BenchRepDataModule
        Configured BenchRep datamodule.

    Raises
    ------
    ValueError
        If ``stage`` is neither ``"training"`` nor ``"prediction"``.
    """
    run_log = get_run_logger()

    if stage not in {"training", "prediction"}:
        raise ValueError(
            "stage must be either 'training' or 'prediction', "
            f"got {stage!r}."
        )

    dm = _instantiate_datamodule(
        datamodule_config=datamodule_config,
        train_dataset=dataset if stage == "training" else None,
        predict_dataset=dataset if stage == "prediction" else None,
        training_pipeline=training_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        seed=seed,
    )

    run_log.info(
        "Built datamodule: stage=%s, dataset=%s, datamodule=%s",
        stage,
        type(dataset).__name__,
        type(dm).__name__,
    )

    return dm


def build_dataset(
    *,
    dataset_config: SupportedDatasetConfig,
) -> BaseDataset:
    """Build a registered BenchRep dataset from validated configuration.

    The dataset name is resolved through the dataset registry. Typed built-in
    parameters or arbitrary custom parameters are converted to constructor
    keyword arguments before the dataset class is instantiated.

    Parameters
    ----------
    dataset_config:
      Validated built-in or custom dataset configuration containing the
      registered dataset name and its constructor parameters.

    Returns
    -------
    BaseDataset
      Instantiated BenchRep-compatible dataset.

    Raises
    ------
    KeyError
      If the configured dataset is not registered.
    TypeError
      If the registered dataset does not produce a ``BaseDataset`` instance.
    """
    run_log = get_run_logger()

    dataset_name = normalize_name(
        dataset_config.name,
        field_name="config.dataset.name",
    )
    dataset_class = DATASETS.get(dataset_name)

    raw_params = dataset_config.params
    if isinstance(raw_params, BaseModel):
        dataset_params = raw_params.model_dump(mode="python")
    else:
        dataset_params = dict(raw_params)

    run_log.info("Building dataset: dataset=%s", dataset_name)

    dataset = dataset_class(**dataset_params)

    if not isinstance(dataset, BaseDataset):
        raise TypeError(
            f"Registered dataset {dataset_name!r} must produce a BaseDataset "
            f"instance, got {type(dataset).__name__}."
        )

    run_log.info(
        "Built dataset: dataset=%s, class=%s, samples=%d",
        dataset_name,
        type(dataset).__name__,
        len(dataset),
    )

    return dataset


def build_transform_pipelines(
    transform_configs: Sequence[TransformConfig],
) -> TransformPipelineBundle:
    """Build training and preprocessing pipelines from ordered transform configs."""

    indexed_configs = tuple(enumerate(transform_configs))

    training_steps = [
        _build_transform_step(
            transform_config,
            index=index,
        )
        for index, transform_config in indexed_configs
    ]

    preprocessing_steps = [
        _build_transform_step(
            transform_config,
            index=index,
        )
        for index, transform_config in indexed_configs
        if transform_config.category == "preprocessing"
    ]

    return TransformPipelineBundle(
        training=TransformPipeline(training_steps),
        preprocessing=TransformPipeline(preprocessing_steps),
    )


def _build_transform_step(
    transform_config: TransformConfig,
    *,
    index: int,
) -> TransformStep:
    transform_name = normalize_name(
        transform_config.name,
        field_name=f"config.transforms[{index}].name",
    )

    transform_factory = TRANSFORMS.get(transform_name)
    transform = transform_factory(**dict(transform_config.params))

    return TransformStep(
        name=transform_name,
        transform=transform,
    )


def _instantiate_datamodule(
    *,
    datamodule_config: DataModuleConfig,
    seed: int | None = None,
    train_dataset: Any | None = None,
    val_dataset: Any | None = None,
    test_dataset: Any | None = None,
    predict_dataset: Any | None = None,
    training_pipeline: TransformPipeline | None = None,
    preprocessing_pipeline: TransformPipeline | None = None,
) -> BenchRepDataModule:
    datamodule_params = datamodule_config.model_dump()

    # Resolve "auto" to pin CPU memory only when CUDA is available.
    if datamodule_params.get("pin_memory") == "auto":
        datamodule_params["pin_memory"] = torch.cuda.is_available()

    return BenchRepDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_dataset=predict_dataset,
        training_pipeline=training_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        seed=seed,
        **datamodule_params,
    )
