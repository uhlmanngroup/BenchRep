from typing import Any

import pytest
import torch

from benchrep.architecture.data import BenchRepDataModule
from benchrep.assembly.builders.data_builder import _instantiate_datamodule
from benchrep.assembly.resolvers.training_config_resolver import (
    _resolve_datamodule_config,
)
from benchrep.assembly.schemas import TrainingDataModuleConfig
from tests.fixtures.datasets import TinySyntheticDataset


def _collate_sample_ids(samples: list[dict[str, Any]]) -> list[str]:
    return [sample["sample_id"] for sample in samples]


@pytest.mark.parametrize(
    ("num_workers", "prefetch_factor", "expected_prefetch"),
    [
        (0, None, None),
        (2, None, 2),
        (2, 3, 3),
    ],
)
def test_loader_options_reach_every_dataloader(
    num_workers: int,
    prefetch_factor: int | None,
    expected_prefetch: int | None,
) -> None:
    generator = torch.Generator().manual_seed(137)
    config = TrainingDataModuleConfig(
        batch_size=2,
        val_fraction=0.0,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        dataloader_kwargs={
            "collate_fn": _collate_sample_ids,
            "generator": generator,
            "timeout": 0 if num_workers == 0 else 5,
        },
    )
    resolved = _resolve_datamodule_config(
        config,
        datamodule_overridden=False,
    )
    assert resolved is not None

    dataset = TinySyntheticDataset(n_samples=5)
    datamodule = _instantiate_datamodule(
        datamodule_config=resolved,
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        predict_dataset=dataset,
    )
    datamodule.setup("fit")

    loaders = [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
        datamodule.predict_dataloader(),
    ]

    for index, loader in enumerate(loaders):
        assert loader is not None
        assert loader.batch_size == 2
        assert loader.num_workers == num_workers
        assert loader.prefetch_factor == expected_prefetch
        assert loader.persistent_workers == (num_workers > 0)
        assert loader.collate_fn is _collate_sample_ids
        assert loader.generator is generator
        assert loader.timeout == (0 if num_workers == 0 else 5)

        # Only training may discard the incomplete final batch.
        assert loader.drop_last == (index == 0)
        assert len(loader) == (2 if index == 0 else 3)

        # Exercise the callable without spawning multiprocessing workers.
        if num_workers == 0:
            batches = list(loader)
            assert len(batches[0]) == 2
            assert all(
                sample_id.startswith("sample_")
                for batch in batches
                for sample_id in batch
            )


@pytest.mark.parametrize(
    "reserved",
    [
        "dataset",
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "drop_last",
        "prefetch_factor",
    ],
)
def test_training_resolver_rejects_reserved_loader_kwargs(
    reserved: str,
) -> None:
    config = TrainingDataModuleConfig(
        num_workers=0,
        pin_memory=False,
        dataloader_kwargs={reserved: None},
    )

    with pytest.raises(ValueError, match="BenchRep-controlled arguments"):
        _resolve_datamodule_config(
            config,
            datamodule_overridden=False,
        )


@pytest.mark.parametrize("pin_memory", [False, True, "auto"])
def test_training_resolver_rejects_unknown_loader_kwargs(
    pin_memory: bool | str,
) -> None:
    config = TrainingDataModuleConfig(
        pin_memory=pin_memory,
        dataloader_kwargs={"not_a_dataloader_argument": True},
    )

    with pytest.raises(ValueError, match="not_a_dataloader_argument"):
        _resolve_datamodule_config(
            config,
            datamodule_overridden=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 1},
        {"multiprocessing_context": "spawn"},
    ],
)
def test_training_resolver_rejects_worker_only_kwargs_without_workers(
    kwargs: dict[str, Any],
) -> None:
    config = TrainingDataModuleConfig(
        num_workers=0,
        dataloader_kwargs=kwargs,
    )

    with pytest.raises(ValueError, match="num_workers"):
        _resolve_datamodule_config(
            config,
            datamodule_overridden=False,
        )


def test_external_datamodule_skips_loader_kwargs_resolution() -> None:
    config = TrainingDataModuleConfig(
        dataloader_kwargs={"not_a_dataloader_argument": True},
    )

    resolved = _resolve_datamodule_config(
        config,
        datamodule_overridden=True,
    )

    assert resolved is config


def test_schema_rejects_explicit_prefetch_without_workers() -> None:
    with pytest.raises(ValueError, match="prefetch_factor"):
        TrainingDataModuleConfig(
            num_workers=0,
            prefetch_factor=2,
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"prefetch_factor": 2}, "prefetch_factor"),
        (
            {"dataloader_kwargs": {"batch_size": 8}},
            "BenchRep-controlled arguments",
        ),
        ({"dataloader_kwargs": {"timeout": 1}}, "num_workers"),
    ],
)
def test_direct_datamodule_construction_validates_loader_options(
    options: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BenchRepDataModule(
            train_dataset=TinySyntheticDataset(n_samples=5),
            val_fraction=0.0,
            num_workers=0,
            **options,
        )


def test_datamodule_copies_caller_loader_kwargs() -> None:
    kwargs = {"timeout": 0}
    datamodule = BenchRepDataModule(
        train_dataset=TinySyntheticDataset(n_samples=5),
        val_fraction=0.0,
        num_workers=0,
        dataloader_kwargs=kwargs,
    )

    kwargs["timeout"] = 10

    assert datamodule.dataloader_kwargs == {"timeout": 0}
    datamodule.setup("fit")
    assert datamodule.train_dataloader().timeout == 0
