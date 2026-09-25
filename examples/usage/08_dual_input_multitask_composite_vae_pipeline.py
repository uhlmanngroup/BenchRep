"""Run a dual-input, multitask Composite VAE pipeline on enriched MNIST.

Registers a custom dataset supplying morphology and skeleton images, digit
labels, stroke-width targets, and simulated acquisition-batch metadata.

Registers a two-input concatenation head that combines encoder features
before variational parameterization. The model reconstructs both images
and predicts digit identity and stroke width.

Run from the repository root:

    python examples/usage/08_dual_input_multitask_composite_vae_pipeline.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from torch import nn
from scipy import ndimage
from skimage.morphology import skeletonize
from torchvision.datasets import MNIST

from benchrep.architecture.data import BaseDataset

from benchrep import (
    inspect_registry,
    train_composite,
    predict_composite,
    evaluate,
)

from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentPort,
    ComponentTensorResult,
)
from benchrep.assembly.registries import DATASETS, HEADS


# Resolve config paths relative to this script.
CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "08_dual_input_multitask_composite_vae_pipeline"
)

# Custom dataset constants
Split = Literal["train", "test"]
DATASET_CONSTRUCTION_SEED: Final = 137
N_ACQUISITION_BATCHES: Final = 10
CHANNEL_NAMES: Final = (
    "morphology",
    "skeleton",
)

IMAGE_SIZE: Final = 28


@dataclass(frozen=True, slots=True)
class AcquisitionBatchProfile:
    """Nuisance parameters associated with one simulated acquisition batch."""

    intensity_gain: float
    intensity_offset: float
    background_amplitude: float
    background_angle_radians: float
    noise_standard_deviation: float
    blur_sigma: float


def build_acquisition_batch_profiles() -> tuple[
    AcquisitionBatchProfile, ...
]:
    """Create reproducible acquisition conditions with independently varied effects."""

    rng = np.random.default_rng(DATASET_CONSTRUCTION_SEED)
    n_batches = N_ACQUISITION_BATCHES

    # Intensity multiplication for the whole img, affects foreground
    intensity_gains = rng.permutation(np.linspace(0.82, 1.18, n_batches))
    # Intensity addition for the whole img, affects foreground and background
    intensity_offsets = rng.permutation(np.linspace(-0.05, 0.05, n_batches))
    # Strength of the uneven background added across the img
    background_amplitudes = rng.permutation(
        np.linspace(0.00, 0.12, n_batches)
    )
    # Direction in which the added background gets brighter
    background_angles = rng.permutation(
        np.linspace(
            0.0,
            2.0 * np.pi,
            n_batches,
            endpoint=False,
        )
    )
    # Std dev of the px-wise gaussian noise
    noise_levels = rng.permutation(np.linspace(0.005, 0.05, n_batches))
    # Sigma for the gaussian blur, in px
    blur_sigmas = rng.permutation(np.linspace(0.0, 0.75, n_batches))

    return tuple(
        AcquisitionBatchProfile(
            intensity_gain=float(intensity_gains[batch_id]),
            intensity_offset=float(intensity_offsets[batch_id]),
            background_amplitude=float(background_amplitudes[batch_id]),
            background_angle_radians=float(background_angles[batch_id]),
            noise_standard_deviation=float(noise_levels[batch_id]),
            blur_sigma=float(blur_sigmas[batch_id]),
        )
        for batch_id in range(n_batches)
    )


def assign_acquisition_batches(
    labels: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Assign each image once while balancing every digit across batches."""

    batch_ids = torch.empty_like(labels)
    generator = torch.Generator().manual_seed(seed)

    for label in torch.unique(labels, sorted=True):
        label_indices = torch.where(labels == label)[0]
        shuffled_indices = label_indices[
            torch.randperm(label_indices.numel(), generator=generator)
        ]
        batch_ids[shuffled_indices] = (
            torch.arange(label_indices.numel()) % N_ACQUISITION_BATCHES
        )

    return batch_ids


ACQUISITION_BATCH_PROFILES: Final = build_acquisition_batch_profiles()


def derive_skeleton_and_stroke_width(
    morphology: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    binary_mask = morphology.numpy() >= 0.2
    skeleton = skeletonize(binary_mask)

    distance_map = ndimage.distance_transform_edt(binary_mask)
    centerline_radii = distance_map[skeleton]
    if centerline_radii.size == 0:
        raise ValueError(
            "Cannot estimate stroke width because the thresholded image "
            "has no skeleton pixels."
        )

    # Approximate stroke width along the skeleton
    mean_stroke_width = float(np.mean(2.0 * centerline_radii - 1.0))

    return torch.from_numpy(skeleton.astype(np.float32)), mean_stroke_width


def build_background_gradient(
    profile: AcquisitionBatchProfile,
) -> torch.Tensor:
    coordinates = torch.linspace(-1.0, 1.0, IMAGE_SIZE)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")

    cos_angle = math.cos(profile.background_angle_radians)
    sin_angle = math.sin(profile.background_angle_radians)

    projection = cos_angle * x + sin_angle * y
    projection /= abs(cos_angle) + abs(sin_angle)

    # Rescale from [-1, 1] to [0, background_amplitude]
    return profile.background_amplitude * 0.5 * (projection + 1.0)


BACKGROUND_GRADIENTS: Final = tuple(
    build_background_gradient(profile)
    for profile in ACQUISITION_BATCH_PROFILES
)


def apply_acquisition_effects(
    channels: torch.Tensor,
    *,
    batch_id: int,
    seed: int,
) -> torch.Tensor:
    profile = ACQUISITION_BATCH_PROFILES[batch_id]

    blurred = ndimage.gaussian_filter(
        channels.numpy(),
        sigma=(0.0, profile.blur_sigma, profile.blur_sigma),
    )
    image = torch.from_numpy(blurred)

    image = (
        profile.intensity_gain * image
        + profile.intensity_offset
        + BACKGROUND_GRADIENTS[batch_id]
    )

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        image.shape,
        dtype=image.dtype,
        generator=generator,
    )

    image += profile.noise_standard_deviation * noise
    return image.clamp(0.0, 1.0)


class EnrichedMNISTDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        split: Split = "train",
        download: bool = False,
    ) -> None:
        super().__init__()

        if split not in ("train", "test"):
            raise ValueError(
                f"split must be 'train' or 'test', got {split!r}."
            )

        self.split = split
        self.dataset = MNIST(
            root=root,
            train=split == "train",
            download=download,
        )

        split_seed = (
            DATASET_CONSTRUCTION_SEED
            if split == "train"
            else DATASET_CONSTRUCTION_SEED + 1_000_000
        )

        self.batch_ids = assign_acquisition_batches(
            self.dataset.targets,
            seed=split_seed,
        )

        self.images, self.mean_stroke_widths = self._prepare_images(
            split_seed=split_seed,
        )

    def _prepare_images(
        self,
        *,
        split_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.empty(
            (len(self.dataset), len(CHANNEL_NAMES), IMAGE_SIZE, IMAGE_SIZE),
            dtype=torch.uint8,
        )
        mean_stroke_widths = torch.empty(
            len(self.dataset),
            dtype=torch.float32,
        )

        for index in range(len(self.dataset)):
            morphology = self.dataset.data[index].to(torch.float32) / 255.0

            skeleton, mean_stroke_width = derive_skeleton_and_stroke_width(
                morphology
            )

            channels = torch.stack(
                (morphology, skeleton),
                dim=0,
            )
            channels = apply_acquisition_effects(
                channels,
                batch_id=int(self.batch_ids[index]),
                seed=split_seed + len(self.dataset) + index,
            )

            images[index] = (255.0 * channels).round().to(torch.uint8)
            mean_stroke_widths[index] = mean_stroke_width

        return images, mean_stroke_widths

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        label = int(self.dataset.targets[index])
        batch_id = int(self.batch_ids[index])

        sample = {
            "morphology": self.images[index, 0:1].clone(),
            "skeleton": self.images[index, 1:2].clone(),
            "digit_label": torch.tensor(label, dtype=torch.long),
            "mean_stroke_width": (
                self.mean_stroke_widths[index].reshape(1).clone()
            ),
            "sample_id": f"{self.split}_{index:05d}",
            "acquisition_batch": f"batch_{batch_id:02d}",
        }
        return self.validate_sample(sample)


class ConcatenationHead(nn.Module):
    """Concatenate morphology and skeleton feature vectors.

    Inputs have shapes [batch, morphology_features] and
    [batch, skeleton_features]. The output retains the batch dimension
    and combines both feature dimensions.
    """

    def forward(
        self,
        *,
        morphology: torch.Tensor,
        skeleton: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((morphology, skeleton), dim=1)


def main() -> None:
    # Register external dataset
    DATASETS.register(
        "enriched_mnist",
        EnrichedMNISTDataset,
    )

    # Register external head
    HEADS.register(
        "concatenate",
        ArchitectureComponent(
            component=ConcatenationHead,
            runtime_inputs=(
                ComponentPort(
                    name="morphology",
                    supported_structures=("vector",),
                ),
                ComponentPort(
                    name="skeleton",
                    supported_structures=("vector",),
                ),
            ),
            runtime_result=ComponentTensorResult(
                supported_structures=("vector",),
            ),
        ),
    )

    print("\n=== External components ===")
    inspect_registry("dataset", "enriched_mnist")
    inspect_registry("head", "concatenate")

    print("\n=== Training ===")
    training_result = train_composite(
        config_path=CONFIG_DIR / "training.yaml",
    )
    print(f"Training manifest: {training_result.manifest_path}")

    print("\n=== Prediction ===")
    prediction_result = predict_composite(
        config_path=CONFIG_DIR / "prediction.yaml",
        training_manifest_path=training_result.manifest_path,
    )
    print(f"Prediction manifest: {prediction_result.manifest_path}")

    print("\n=== Evaluation ===")
    evaluation_result = evaluate(
        config_path=CONFIG_DIR / "evaluation.yaml",
        prediction_manifest_path=prediction_result.manifest_path,
    )
    print(f"Evaluation manifest: {evaluation_result.manifest_path}")

    print("\n=== Pipeline complete ===")
    print(f"Training outputs:   {training_result.run_context.output_dir}")
    print(f"Prediction outputs: {prediction_result.run_context.output_dir}")
    print(f"Evaluation outputs: {evaluation_result.run_context.output_dir}")


if __name__ == "__main__":
    main()