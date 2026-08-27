from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader, Dataset

from tests.fixtures.datasets import CompatibleAutoencoderBatchDataset


class ExternalDataModule(L.LightningDataModule):
    """Minimal external datamodule for exercising dataset batch contracts."""

    def __init__(
        self,
        *,
        train_dataset: Dataset,
        val_dataset: Dataset,
        predict_dataset: Dataset,
        batch_size: int = 8,
    ) -> None:
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


class ParameterizedExternalDataModule(ExternalDataModule):
    """External datamodule constructible entirely from config parameters."""

    def __init__(
        self,
        *,
        train_samples: int = 24,
        val_samples: int = 8,
        predict_samples: int = 32,
        batch_size: int = 8,
        seed: int = 137,
    ) -> None:
        super().__init__(
            train_dataset=CompatibleAutoencoderBatchDataset(
                n_samples=train_samples,
                seed=seed,
            ),
            val_dataset=CompatibleAutoencoderBatchDataset(
                n_samples=val_samples,
                seed=seed + 1,
            ),
            predict_dataset=CompatibleAutoencoderBatchDataset(
                n_samples=predict_samples,
                seed=seed + 2,
            ),
            batch_size=batch_size,
        )

        self.sample_counts = {
            "train": train_samples,
            "validation": val_samples,
            "prediction": predict_samples,
        }
        self.seed = seed
