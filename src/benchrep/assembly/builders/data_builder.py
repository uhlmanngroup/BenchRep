from __future__ import annotations

from typing import Any

import torch

from benchrep.records import get_run_logger
from benchrep.architecture.data import DataModule
from benchrep.assembly.schemas import (
    DataModuleConfig,
    DatasetConfig,
    TransformConfig
)
from benchrep.assembly.config_utils import normalize_name
from benchrep.assembly.registry import DATASETS, TRANSFORMS


def build_datamodule(
        dataset_config: DatasetConfig,
        datamodule_config: DataModuleConfig,
        seed: int | None = None) -> DataModule:
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
    dataset_config:
        Validated dataset config object containing dataset section.
    datamodule_config:
        Validated datamodule config object containing datamodule section.
    seed:
        Optional random seed passed to the DataModule for reproducible
        train/validation splitting.

    Returns
    -------
    DataModule
        Instantiated DataModule containing the datasets requested by the builder.
    """
    run_log = get_run_logger()

    dataset_name = normalize_name(
        dataset_config.name,
        field_name="config.data.dataset.name",
    )

    run_log.info("Building dataset...: %s", dataset_name)

    if dataset_name == "mnist":
        train_dataset, test_dataset = _build_mnist_datasets(dataset_config)
        dm = _instantiate_datamodule(
            datamodule_config=datamodule_config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            seed=seed,
        )

        run_log.info(
            "Built datamodule: dataset=%s, datamodule=%s",
            dataset_name,
            type(dm).__name__,
        )

        return dm

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
    seed: int | None = None,
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
        field_name="config.data.dataset.transform.name",
    )

    transform_class = TRANSFORMS.get(transform_name)
    transform_params = dict(transform_config.params)

    return transform_class(**transform_params)