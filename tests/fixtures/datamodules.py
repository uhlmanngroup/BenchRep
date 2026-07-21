from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader, Dataset


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