from __future__ import annotations

from typing import Any

import lightning as L
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split


class DataModule(L.LightningDataModule):
    """Generic LightningDataModule for BenchRep-compatible datasets.

    Datasets are expected to return dictionary samples following the internal
    contract used by BenchRep models and enforced by BaseDataset, with at least key ``"x"``.

    The DataModule supports training-only runs, training with an explicit validation
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
    """

    def __init__(
        self,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        predict_dataset: Dataset | None = None,
        batch_size: int = 64,
        val_fraction: float = 0.1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        seed: int = 137,
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

        self._original_train_dataset = train_dataset
        self._provided_val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset

        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.seed = seed

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        # Lightning may call setup multiple times. If setup already prepared the final
        # training dataset, do not split or reassign train/val datasets again.
        if self.train_dataset is not None:
            return

        # train_dataset is optional so the same DataModule can support test-only or
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
            dataset_size = len(self._original_train_dataset)
            val_size = int(dataset_size * self.val_fraction)
            train_size = dataset_size - val_size

            if val_size == 0:
                raise ValueError(
                    f"val_fraction={self.val_fraction} produced an empty validation set "
                    f"for dataset of size {dataset_size}."
                )

            generator = Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                self._original_train_dataset,
                [train_size, val_size],
                generator=generator,
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
        )