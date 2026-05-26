from __future__ import annotations

from typing import Any

import torch

from benchrep.architecture.data import DataModule
from benchrep.assembly.schemas import (
    DataConfig,
    DataModuleConfig,
    DatasetConfig,
    TransformConfig
)
from benchrep.assembly.config_utils import normalize_name
from benchrep.assembly.registry import DATASETS, TRANSFORMS


def build_datamodule(data_config: DataConfig) -> DataModule:
    """Build a DataModule from the validated DataConfig object.

    This is the public data builder. It translates the ``data`` section of a loaded
    config into concrete dataset objects and a BenchRep ``DataModule``.

    For the built-in MNIST toy dataset, this builder handles MNIST's predefined
    train/test split by creating both ``MNISTDataset(train=True)`` and
    ``MNISTDataset(train=False)`` from the same dataset config. Future dataset types
    may use different construction logic internally, but should still return a
    ``DataModule`` through this function.

    Parameters
    ----------
    data_config:
        Validated data config object containing dataset and datamodule sections.

    Returns
    -------
    DataModule
        Instantiated DataModule containing the datasets requested by the builder.
    """
    dataset_config = data_config.dataset
    datamodule_config = data_config.datamodule

    dataset_name = normalize_name(
        dataset_config.name,
        field_name="config.data.dataset.name",
    )

    if dataset_name == "mnist":
        train_dataset, test_dataset = _build_mnist_datasets(dataset_config)
        return _instantiate_datamodule(
            datamodule_config=datamodule_config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

    raise ValueError(
        f"Unsupported dataset name {dataset_name!r}. "
        f"Available options: {DATASETS.keys()}."
    )


def _build_mnist_datasets(dataset_config: DatasetConfig) -> tuple[Any, Any]:
    # MNIST has a known train/test split, so the builder creates both datasets
    # from the same dataset config.
    if dataset_config.root is None:
        raise ValueError("MNIST dataset requires `data.dataset.root`.")

    dataset_class = DATASETS.get("mnist")

    root = dataset_config.root
    download = bool(dataset_config.download)
    transform = _build_transform(dataset_config.transform)

    train_dataset = dataset_class(
        root=root,
        train=True,
        transform=transform,
        download=download,
    )

    test_dataset = dataset_class(
        root=root,
        train=False,
        transform=transform,
        download=download,
    )

    return train_dataset, test_dataset


def _instantiate_datamodule(
    datamodule_config: DataModuleConfig,
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
        **datamodule_params,
    )


def _build_transform(transform_config: TransformConfig | None) -> Any:
    if transform_config is None:
        return None

    transform_name = normalize_name(
        transform_config.name,
        field_name="config.data.dataset.transform.name",
    )

    transform_class = TRANSFORMS.get(transform_name)
    transform_params = dict(transform_config.params)

    return transform_class(**transform_params)