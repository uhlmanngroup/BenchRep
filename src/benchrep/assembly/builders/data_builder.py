from __future__ import annotations

from typing import Any

import torch

from benchrep.architecture.data import DataModule
from benchrep.assembly.config_utils import (
    get_optional_section,
    get_required_section,
    get_required_value,
    normalize_name,
    require_mapping,
)
from benchrep.assembly.registry import DATASETS, TRANSFORMS


def build_datamodule(data_config: dict[str, Any]) -> DataModule:
    """Build a DataModule from data config.

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
        Data config dictionary. Expected to contain required ``"dataset"`` and
        ``"datamodule"`` sections.

    Returns
    -------
    DataModule
        Instantiated DataModule containing the datasets requested by the builder.
    """
    data_config = require_mapping(data_config, "data_config")

    dataset_config = get_required_section(data_config, "dataset")
    datamodule_config = get_required_section(data_config, "datamodule")

    dataset_name = normalize_name(
        get_required_value(dataset_config, "name"),
        field_name="data_config['dataset']['name']",
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


def _build_mnist_datasets(dataset_config: dict[str, Any]) -> tuple[Any, Any]:
    # MNIST has a known train/test split, so the builder creates both datasets
    # from the same dataset config.
    dataset_class = DATASETS.get("mnist")

    root = get_required_value(dataset_config, "root")
    download = dataset_config.get("download", False)
    transform = _build_transform(dataset_config.get("transform"))

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
    datamodule_config: dict[str, Any],
    train_dataset: Any | None = None,
    val_dataset: Any | None = None,
    test_dataset: Any | None = None,
    predict_dataset: Any | None = None,
) -> DataModule:
    datamodule_params = dict(datamodule_config)

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


def _build_transform(transform_config: dict[str, Any] | None) -> Any:
    if transform_config is None:
        return None

    transform_config = require_mapping(transform_config, "transform config")

    transform_name = normalize_name(
        get_required_value(transform_config, "name"),
        field_name="transform config['name']",
    )

    transform_class = TRANSFORMS.get(transform_name)
    transform_params = get_optional_section(transform_config, "params")

    return transform_class(**transform_params)