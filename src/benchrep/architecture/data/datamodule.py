from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
from inspect import Parameter, signature
from typing import Any

import lightning as L
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split

from benchrep.architecture.data.datasets import TransformedDataset
from benchrep.architecture.data.transforms import TransformPipeline


_RESERVED_DATALOADER_KWARGS = frozenset({
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
})


def validate_dataloader_kwargs(
    dataloader_kwargs: Mapping[str, Any],
    *,
    num_workers: int,
) -> None:
    """Validate additional kwargs against BenchRep ownership and PyTorch."""
    if any(not isinstance(key, str) for key in dataloader_kwargs):
        raise ValueError("dataloader_kwargs keys must be strings.")

    reserved = _RESERVED_DATALOADER_KWARGS.intersection(dataloader_kwargs)
    if reserved:
        raise ValueError(
            "dataloader_kwargs cannot override BenchRep-controlled arguments: "
            f"{', '.join(sorted(reserved))}."
        )

    accepted = {
        name
        for name, parameter in signature(DataLoader).parameters.items()
        if parameter.kind in (
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        )
    }
    unknown = set(dataloader_kwargs) - accepted
    if unknown:
        raise ValueError(
            "dataloader_kwargs contains arguments not accepted by the installed "
            f"DataLoader: {', '.join(sorted(unknown))}."
        )

    if num_workers == 0:
        if dataloader_kwargs.get("multiprocessing_context") is not None:
            raise ValueError(
                "dataloader_kwargs.multiprocessing_context requires num_workers > 0."
            )

        if dataloader_kwargs.get("timeout", 0) != 0:
            raise ValueError(
                "dataloader_kwargs.timeout must be 0 when num_workers=0."
            )


class BenchRepDataModule(L.LightningDataModule):
    """LightningDataModule for BenchRep-compatible datasets.

    Datasets must return dictionary samples following the generic
    `BaseDataset` contract. Model-specific required fields are validated by the
    consuming canonical or Composite model.

    The BenchRepDataModule supports training-only runs, training with an explicit validation
    dataset, training with a validation split from the training dataset, test-only
    runs, prediction-only runs, and combined test/prediction use.

    Parameters
    ----------
    train_dataset:
        Optional dataset used for training. Required for training runs, but not for
        test-only or prediction-only use.
    val_dataset:
        Optional validation dataset. If not provided and ``val_fraction > 0``,
        the training dataset is split into train/validation subsets.
    test_dataset:
        Optional test dataset.
    predict_dataset:
        Optional dataset used for prediction/inference with ``Trainer.predict()``.
    training_pipelines:
        Ordered transform pipeline applied to training samples.
    validation_pipelines:
        Ordered transform pipeline applied to validation and test samples. It
        also acts as the prediction fallback when no explicit prediction
        pipeline is supplied.
    prediction_pipelines:
        Optional explicit transform pipeline applied to prediction samples.
        When omitted, ``validation_pipeline`` is used.
    batch_size:
        Number of samples per batch.
    val_fraction:
        Fraction of ``train_dataset`` used for validation when ``val_dataset`` is
        not provided.
    num_workers:
        Number of worker processes used by each DataLoader.
    pin_memory:
        Whether DataLoaders should use pinned memory.
    persistent_workers:
        Whether DataLoader workers should persist across epochs. Requires
        ``num_workers > 0``.
    drop_last:
        Whether to drop the last incomplete training batch.
    seed:
        Random seed used for train/validation splitting.
    prefetch_factor:
        Number of batches prefetched per worker. None uses PyTorch's default:
        two batches per worker when multiprocessing is enabled. An explicit
        value requires ``num_workers > 0``.
    dataloader_kwargs:
        Additional keyword arguments passed to every DataLoader. Cannot
        override BenchRep-controlled arguments.
    """

    def __init__(
        self,
        train_dataset: Dataset[dict[str, Any]] | None = None,
        val_dataset: Dataset[dict[str, Any]] | None = None,
        test_dataset: Dataset[dict[str, Any]] | None = None,
        predict_dataset: Dataset[dict[str, Any]] | None = None,
        training_pipelines: Sequence[TransformPipeline] | None = None,
        validation_pipelines: Sequence[TransformPipeline] | None = None,
        prediction_pipelines: Sequence[TransformPipeline] | None = None,
        batch_size: int = 64,
        val_fraction: float = 0.1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
        prefetch_factor: int | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if train_dataset is None and val_dataset is not None:
            raise ValueError("val_dataset can only be provided when train_dataset is provided.")

        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")

        if train_dataset is None and val_fraction > 0:
            raise ValueError("val_fraction must be 0 when train_dataset is not provided.")

        if train_dataset is None and test_dataset is None and predict_dataset is None:
            raise ValueError(
                "At least one of train_dataset, test_dataset, or predict_dataset must be provided."
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        if val_dataset is not None and val_fraction > 0:
            raise ValueError(
                "val_fraction must be 0 when val_dataset is provided. "
                "Pass either an explicit val_dataset or request a train/val split."
            )
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0.")

        if prefetch_factor is not None:
            if (
                isinstance(prefetch_factor, bool)
                or not isinstance(prefetch_factor, int)
                or prefetch_factor <= 0
            ):
                raise ValueError(
                    "prefetch_factor must be a positive integer or None."
                )

            if num_workers == 0:
                raise ValueError("prefetch_factor requires num_workers > 0.")

        loader_kwargs = (
            {} if dataloader_kwargs is None else dict(dataloader_kwargs)
        )
        validate_dataloader_kwargs(
            loader_kwargs,
            num_workers=num_workers,
        )

        self._original_train_dataset = train_dataset
        self._provided_val_dataset = val_dataset

        self.training_pipelines = tuple(training_pipelines or ())
        self.validation_pipelines = tuple(validation_pipelines or ())
        self.prediction_pipelines = (
            tuple(prediction_pipelines)
            if prediction_pipelines is not None
            else self.validation_pipelines
        )

        # Test inputs follow validation-time processing by default.
        self.test_dataset = _wrap_with_pipelines(
            test_dataset,
            self.validation_pipelines,
        )
        self.predict_dataset = _wrap_with_pipelines(
            predict_dataset,
            self.prediction_pipelines,
        )

        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.dataloader_kwargs = loader_kwargs

        self.train_dataset: Dataset[dict[str, Any]] | None = None
        self.val_dataset: Dataset[dict[str, Any]] | None = None

    def setup(self, stage: str | None = None) -> None:
        # Lightning may call setup multiple times. If setup already prepared the final
        # training dataset, do not split or reassign train/val datasets again.
        if self.train_dataset is not None:
            return

        # train_dataset is optional so the same BenchRepDataModule can support test-only or
        # predict-only runs. In that case, there is no train/val setup to perform.
        if self._original_train_dataset is None:
            return

        # Explicit validation dataset provided; use it as-is.
        if self._provided_val_dataset is not None:
            self.train_dataset = self._original_train_dataset
            self.val_dataset = self._provided_val_dataset

        # No validation requested; use the full training dataset for training.
        elif self.val_fraction == 0:
            self.train_dataset = self._original_train_dataset
            self.val_dataset = None

        # Validation requested as a fraction of the provided training dataset.
        else:
            assert isinstance(self._original_train_dataset, Sized)

            dataset_size = len(self._original_train_dataset)
            val_size = int(dataset_size * self.val_fraction)
            train_size = dataset_size - val_size

            if val_size == 0:
                raise ValueError(
                    f"val_fraction={self.val_fraction} produced an empty validation set "
                    f"for dataset of size {dataset_size}."
                )

            generator = None
            if self.seed is not None:
                generator = Generator().manual_seed(self.seed)

            assert isinstance(self._original_train_dataset, Dataset)
            self.train_dataset, self.val_dataset = random_split(
                self._original_train_dataset,
                [train_size, val_size],
                generator=generator,
            )

        self.train_dataset = _wrap_with_pipelines(
            self.train_dataset,
            self.training_pipelines,
        )
        self.val_dataset = _wrap_with_pipelines(
            self.val_dataset,
            self.validation_pipelines,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "No training dataset is available. Provide train_dataset before calling "
                "train_dataloader()."
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            prefetch_factor=self.prefetch_factor,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            prefetch_factor=self.prefetch_factor,
            **self.dataloader_kwargs,
        )

    def predict_dataloader(self) -> DataLoader | None:
        if self.predict_dataset is None:
            return None

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            prefetch_factor=self.prefetch_factor,
            **self.dataloader_kwargs,
        )


def _wrap_with_pipelines(
    dataset: Dataset[dict[str, Any]] | None,
    pipelines: Sequence[TransformPipeline] | None,
) -> Dataset[dict[str, Any]] | None:
    if dataset is None or not pipelines:
        return dataset

    effective_pipelines = tuple(
        pipeline
        for pipeline in pipelines
        if len(pipeline) > 0
    )

    if not effective_pipelines:
        return dataset

    return TransformedDataset(
        dataset=dataset,
        pipelines=effective_pipelines,
    )
