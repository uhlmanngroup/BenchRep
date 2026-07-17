from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

import torch

from benchrep.records import get_run_logger
from benchrep.architecture.data import BaseDataset, DataModule
from benchrep.assembly.schemas import (
    DataModuleConfig,
    TransformConfig,
    SupportedDatasetConfig,
)
from benchrep.assembly.registries.utils import normalize_name
from benchrep.assembly.registries.core import DATASETS, TRANSFORMS


def build_datamodule(
    *,
    dataset: BaseDataset,
    datamodule_config: DataModuleConfig,
    seed: int | None = None,
    stage: Literal["training", "prediction"],
) -> DataModule:
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
    DataModule
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
      keyword arguments. A configured transform is resolved through the transform
      registry before the dataset class is instantiated.

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
          If the dataset or configured transform is not registered.
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

    transform_config = dataset_params.get("transform")

    if isinstance(transform_config, dict) and "name" in transform_config:
        transform_config = TransformConfig.model_validate(transform_config)

    if isinstance(transform_config, TransformConfig):
        dataset_params["transform"] = _build_transform(transform_config)

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


def _instantiate_datamodule(
    *,
    datamodule_config: DataModuleConfig,
    seed: int | None = None,
    train_dataset: Any | None = None,
    val_dataset: Any | None = None,
    test_dataset: Any | None = None,
    predict_dataset: Any | None = None,
) -> DataModule:
    datamodule_params = datamodule_config.model_dump()

    # Resolve "auto" to pin CPU memory only when CUDA is available.
    if datamodule_params.get("pin_memory") == "auto":
        datamodule_params["pin_memory"] = torch.cuda.is_available()

    return DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_dataset=predict_dataset,
        seed=seed,
        **datamodule_params,
    )


def _build_transform(transform_config: TransformConfig | None) -> Any:
    if transform_config is None:
        return None

    transform_name = normalize_name(
        transform_config.name,
        field_name="config.dataset.transform.name",
    )

    transform_class = TRANSFORMS.get(transform_name)
    transform_params = dict(transform_config.params)

    return transform_class(**transform_params)