from typing import Any

import torch

from benchrep.architecture.data.datasets import BaseDataset


class TinySyntheticDataset(BaseDataset):
    def __init__(
        self,
        n_samples: int = 32,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        n_classes: int = 4,
        n_groups: int = 2,
        signal_strength: float = 0.8,
        noise_std: float = 0.05,
        seed: int = 137,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if n_classes < 1:
            raise ValueError("n_classes must be positive.")
        if n_groups < 1:
            raise ValueError("n_groups must be positive.")

        channels, height, width = image_shape
        generator = torch.Generator().manual_seed(seed)

        self.labels = torch.arange(n_samples) % n_classes
        self.groups = (torch.arange(n_samples) // n_classes) % n_groups
        self.continuous_targets = torch.rand(
            n_samples,
            generator=generator,
        )

        self.images = torch.randn(
            (n_samples, channels, height, width),
            generator=generator,
        ) * noise_std

        for index in range(n_samples):
            label = int(self.labels[index])

            # Encode categorical class as a class-specific vertical region.
            start = label * width // n_classes
            stop = (label + 1) * width // n_classes
            self.images[index, :, :, start:stop] += signal_strength

            # Encode the continuous target in a horizontal region.
            self.images[index, :, -2:, :] += (
                signal_strength * self.continuous_targets[index]
            )

        self.images.clamp_(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        label = int(self.labels[index])
        group = int(self.groups[index])

        sample = {
            "x": self.images[index],
            "label": label,
            "sample_id": f"sample_{index:04d}",
            "metadata": {
                "label_str": f"class_{label}",
                "continuous_target": float(self.continuous_targets[index]),
                "group": f"group_{group}",
            },
        }

        return self.validate_sample(sample)