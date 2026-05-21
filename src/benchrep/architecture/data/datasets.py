from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class BaseDataset(Dataset, ABC):
    """Base interface for BenchRep-compatible datasets.

    Subclasses should follow the standard PyTorch Dataset API and implement
    ``__len__`` and ``__getitem__``.

    Each sample returned by ``__getitem__`` must be a dictionary containing at
    least the key ``"x"``. ``sample["x"]`` must be the input tensor consumed by
    models.

    Optional keys may include labels, identifiers, metadata, paths, coordinates,
    or any other information needed by downstream workflows.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample as a dictionary containing at least key ``"x"``."""
        raise NotImplementedError

    @staticmethod
    def validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
        """Validate and return a sample following the BenchRep dataset contract."""
        if not isinstance(sample, dict):
            raise TypeError(
                "Dataset samples must be dictionaries containing at least key 'x'. "
                f"Got {type(sample).__name__}."
            )

        if "x" not in sample:
            raise KeyError(
                "Dataset sample must contain key 'x'. "
                f"Available keys: {tuple(sample.keys())}."
            )

        if not isinstance(sample["x"], torch.Tensor):
            raise TypeError(
                "sample['x'] must be a torch.Tensor, "
                f"got {type(sample['x']).__name__}."
            )

        return sample


class MNISTDataset(BaseDataset):
    """Wrapper around torchvision MNIST using the BenchRep dataset contract.

    Parameters
    ----------
    root:
        Directory where MNIST data is stored or downloaded.
    train:
        Whether to use the training split.
    transform:
        Optional transform applied to the image.
    target_transform:
        Optional transform applied to the label.
    download:
        Whether to download MNIST if it is not already present.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Any | None = None,
        target_transform: Any | None = None,
        download: bool = False,
    ) -> None:
        super().__init__()

        self.dataset = MNIST(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        x, y = self.dataset[index]

        sample = {
            "x": x,
            "y": y,
        }

        return self.validate_sample(sample)