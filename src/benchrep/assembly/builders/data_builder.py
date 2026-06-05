from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import ConcatDataset

from benchrep.records import get_run_logger
from benchrep.architecture.data import DataModule
from benchrep.assembly.schemas import (
    DataModuleConfig,
    DatasetConfig,
    TransformConfig
)
from benchrep.assembly.registry_utils import normalize_name
from benchrep.assembly.registry import DATASETS, TRANSFORMS


def build_datamodule(
        *,
        dataset_config: DatasetConfig,
        datamodule_config: DataModuleConfig,
        seed: int | None = None,
        stage: str,
        split: str
) -> DataModule:
    """Build a BenchRep DataModule for a given workflow stage and data split.

    This is the public data builder. It translates the validated dataset and
    datamodule config sections, together with the requested workflow ``stage`` and
    data ``split``, into concrete dataset objects and a BenchRep ``DataModule``.

    The ``split`` argument selects which dataset view to instantiate. The ``stage``
    argument determines which LightningDataModule slot receives that dataset. For
    example, ``stage="training"`` assigns the requested split to ``train_dataset``,
    while ``stage="prediction"`` assigns the requested split to
    ``predict_dataset``.

    For the built-in MNIST toy dataset, BenchRep split names are mapped onto
    torchvision MNIST's ``train`` boolean flag. For example, ``split="train"``
    creates ``MNISTDataset(train=True)``, while ``split="test"`` or
    ``split="predict"`` creates ``MNISTDataset(train=False)``. Future dataset
    types may implement different split-selection logic internally, but should
    still return a concrete dataset object through this builder.

    Parameters
    ----------
    dataset_config:
        Validated dataset config object describing the dataset family/source and
        its construction parameters, such as dataset name, root path, download flag,
        and transform.
    datamodule_config:
        Validated datamodule config object describing batching and DataLoader
        behavior, such as batch size, number of workers, pin-memory behavior,
        persistent workers, validation fraction, and drop-last behavior.
    seed:
        Optional random seed passed to the DataModule. This is mainly used for
        reproducible train/validation splitting when the DataModule creates a
        validation subset from a training dataset.
    stage:
        Workflow stage that determines which DataModule dataset slot is populated.
        Currently supported values are ``"training"`` and ``"prediction"``.
    split:
        Dataset split or view to instantiate. For MNIST, currently supported values
        are ``"train"``, ``"test"``, ``"predict"``, and ``"all"``.

    Returns
    -------
    DataModule
        Instantiated BenchRep DataModule containing the requested dataset assigned
        to the appropriate Lightning stage slot.

    Raises
    ------
    ValueError
        If the dataset name, workflow stage, or requested split is unsupported, or
        if required dataset construction parameters are missing.
    """
    run_log = get_run_logger()

    dataset_name = normalize_name(
        dataset_config.name,
        field_name="config.dataset.name",
    )

    run_log.info(
        "Building datamodule: dataset=%s, stage=%s, split=%s",
        dataset_name,
        stage,
        split,
    )

    # Build dataset
    if dataset_name == "mnist":
        dataset = _build_mnist_dataset(
            dataset_config=dataset_config,
            split=split,
        )
    else:
        raise ValueError(
                f"Unsupported dataset name {dataset_name!r}. "
                f"Available options: {DATASETS.keys()}."
            )

    # Build datamodule
    if stage == "training":
        dm = _instantiate_datamodule(
            datamodule_config=datamodule_config,
            train_dataset=dataset,
            seed=seed,
        )
    elif stage == "prediction":
        dm_config = datamodule_config.model_copy(
            update={
                "val_fraction": 0.0,
                "drop_last": False,
            }
        )

        dm = _instantiate_datamodule(
            datamodule_config=dm_config,
            predict_dataset=dataset,
            seed=seed,
        )
    else:
        raise ValueError(
            "Unsupported datamodule stage. Expected one of "
            "{'training', 'prediction'}, "
            f"got {stage!r}."
        )

    run_log.info(
        "Built datamodule: dataset=%s, stage=%s, split=%s, datamodule=%s",
        dataset_name,
        stage,
        split,
        type(dm).__name__,
    )

    return dm


def _build_mnist_dataset(
        dataset_config: DatasetConfig,
        split: str,
) -> Any:
    # MNIST exposes train/test through a boolean flag; map BenchRep split names
    # onto that torchvision API.
    if dataset_config.root is None:
        raise ValueError("MNIST dataset requires `dataset.root`.")

    dataset_class = DATASETS.get("mnist")

    root = dataset_config.root
    download = bool(dataset_config.download)
    transform = _build_transform(dataset_config.transform)

    if split == "train":
        return dataset_class(
            root=root,
            train=True,
            transform=transform,
            download=download,
        )

    if split in {"test", "predict"}:
        return dataset_class(
            root=root,
            train=False,
            transform=transform,
            download=download,
        )

    if split == "all":
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
        return ConcatDataset([train_dataset, test_dataset])

    raise ValueError(
        "Unsupported MNIST split. Expected one of "
        "{'train', 'test', 'predict', 'all'}, "
        f"got {split!r}."
    )


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