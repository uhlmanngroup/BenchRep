from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from benchrep.architecture.data.transforms import TransformPipeline


class BaseDataset(Dataset[dict[str, Any]], ABC):
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


class TransformedDataset(BaseDataset):
    """Wrap a dataset and transform each sample's ``"x"`` tensor.

    The wrapped dataset must return samples following the BenchRep sample
    contract. Only ``sample["x"]`` is transformed; all other sample fields are
    preserved unchanged.

    Notes
    -----
    The sample dictionary is shallow-copied, but ``sample["x"]`` is not cloned.
    An in-place transform may therefore modify tensor storage owned by the
    wrapped dataset.
    """

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        pipeline: TransformPipeline,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.pipeline = pipeline

    def __len__(self) -> int:
        assert isinstance(self.dataset, Sized)

        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.validate_sample(self.dataset[index])

        transformed_sample = {
            **sample,
            "x": self.pipeline(sample["x"]),
        }

        return self.validate_sample(transformed_sample)


class MNISTDataset(BaseDataset):
    """Adapt torchvision MNIST to the BenchRep sample contract.

    Each item is returned as a dictionary containing the image under ``"x"``,
    the digit class under ``"label"``, and the split-local integer index under
    ``"sample_id"``.

    Parameters
    ----------
    root:
        Directory containing or receiving the MNIST files.
    split:
        MNIST split to load. ``"train"`` selects the training set and
        ``"test"`` selects the test set.
    transform:
        Optional callable passed to torchvision MNIST and applied to each image.
        It must produce a tensor satisfying the BenchRep sample contract.
    target_transform:
        Optional callable passed to torchvision MNIST and applied to each label.
    download:
        Whether torchvision should download MNIST when it is unavailable under
        ``root``.

    Notes
    -----
    Without an image transform, torchvision MNIST returns PIL images, which do
    not satisfy the current BenchRep requirement that ``sample["x"]`` be a
    tensor.
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        transform: Any | None = None,
        target_transform: Any | None = None,
        download: bool = False,
    ) -> None:
        super().__init__()

        self.dataset = MNIST(
            root=root,
            train=split == "train",
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
            "label": y,
            "sample_id": index,
        }

        return self.validate_sample(sample)