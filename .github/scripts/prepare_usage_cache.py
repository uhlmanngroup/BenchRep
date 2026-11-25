"""Prepare portable inputs for the current tests/usage suite, without running it.

Update this inventory when the usage suite adds datasets or pretrained weights.
CI hashes this file to version the cache; ordinary code/doc edits do not invalidate it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def prepare(data_dir: Path) -> None:
    import torch
    from torchvision.datasets import CIFAR10, FashionMNIST, MNIST, STL10
    from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, ViT_B_16_Weights

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    # Matches TORCH_HOME=data/torch in the matrix, including Windows.
    torch.hub.set_dir(str(data_dir / "torch" / "hub"))

    for dataset_class, folder in ((MNIST, "mnist"), (FashionMNIST, "fashion_mnist"),
                                  (CIFAR10, "cifar10"), (STL10, "stl10")):
        start = time.monotonic()
        root = data_dir / folder
        print(f"Preparing {dataset_class.__name__} in {root}", flush=True)
        if dataset_class is STL10:
            # The train reader downloads the complete archive, including unlabeled
            # data, without loading all 100,000 unlabeled images into memory.
            dataset_class(root=str(root), split="train", download=True)
            # download=False checks the extracted files, including unlabeled_X.bin.
            dataset_class(root=str(root), split="test", download=False)
        else:
            dataset_class(root=str(root), train=True, download=True)
            dataset_class(root=str(root), train=False, download=False)

        # Keep the real extracted data and metadata. Remove only redundant archives,
        # after both readers above succeed; no user checkout files are involved.
        if dataset_class in (CIFAR10, STL10):
            archives = [root / dataset_class.filename]
        else:
            raw = root / dataset_class.__name__ / "raw"
            archives = [raw / filename for filename, _ in dataset_class.resources]
        for archive in archives:
            archive.unlink(missing_ok=True)
        print(f"Prepared {dataset_class.__name__} in {time.monotonic() - start:.1f}s", flush=True)

    for weights in (ResNet18_Weights.IMAGENET1K_V1, ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                    ViT_B_16_Weights.IMAGENET1K_V1):
        print(f"Preparing {weights}", flush=True)
        weights.get_state_dict(progress=True, check_hash=True)

    size = sum(path.stat().st_size for path in data_dir.rglob("*") if path.is_file())
    print(f"Prepared inputs: {size / 1024**3:.2f} GiB before cache compression", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    prepare(parser.parse_args().data_dir)
