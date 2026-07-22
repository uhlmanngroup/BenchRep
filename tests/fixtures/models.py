from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired
from typing_extensions import TypedDict

import torch
from torch import nn

from benchrep.interfaces.contracts import (
    AutoencoderBatch,
    PredictionMetadata,
    PredictionObsValue,
)
from benchrep.interfaces.models import BenchRepAutoencoderModel


@dataclass(slots=True)
class ExternalAutoencoderPrediction:
    embedding: torch.Tensor
    input: torch.Tensor
    reconstruction: torch.Tensor
    sample_id: PredictionObsValue | None = None
    label: PredictionObsValue | None = None
    metadata: PredictionMetadata | None = None


class CompatibleExternalAutoencoder(BenchRepAutoencoderModel):
    """Minimal user-owned model satisfying BenchRep's external AE contracts."""

    def __init__(
        self,
        latent_dim: int = 8,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.compressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, latent_dim),
        )
        self.reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.compressor(images)
        restored_images = self.reconstructor(representation).reshape_as(images)
        return restored_images, representation

    def training_step(
        self,
        batch: AutoencoderBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        images = batch["x"]
        restored_images, _ = self(images)
        loss = torch.mean((restored_images - images) ** 2)

        self.log(
            "train/loss",
            loss,
            batch_size=images.shape[0],
        )

        return loss

    def validation_step(
        self,
        batch: AutoencoderBatch,
        batch_idx: int,
    ) -> None:
        images = batch["x"]
        restored_images, _ = self(images)
        loss = torch.mean((restored_images - images) ** 2)

        self.log(
            "val/loss",
            loss,
            batch_size=images.shape[0],
        )

    def predict_step(
        self,
        batch: AutoencoderBatch,
        batch_idx: int,
    ) -> ExternalAutoencoderPrediction:
        images = batch["x"]
        restored_images, representation = self(images)

        return ExternalAutoencoderPrediction(
            embedding=representation,
            input=images,
            reconstruction=restored_images,
            sample_id=batch.get("sample_id"),
            label=batch.get("label"),
            metadata=batch.get("metadata"),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
        )


class PrivateImageBatch(TypedDict):
    """Private batch contract shared only by the external model and dataset."""

    image: torch.Tensor
    identifier: NotRequired[PredictionObsValue]
    attributes: NotRequired[PredictionMetadata]


class PrivateBatchExternalAutoencoder(BenchRepAutoencoderModel):
    """External model consuming a private, non-BenchRep batch contract."""

    def __init__(
        self,
        latent_dim: int = 8,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.lr = lr

        self.feature_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.image_network = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.feature_network(image)
        restored = self.image_network(features).reshape_as(image)

        return {
            "features": features,
            "restored": restored,
        }

    def training_step(
        self,
        batch: PrivateImageBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        image = batch["image"]
        result = self(image)
        loss = torch.mean(torch.abs(result["restored"] - image))

        self.log(
            "train/loss",
            loss,
            batch_size=image.shape[0],
        )

        return loss

    def validation_step(
        self,
        batch: PrivateImageBatch,
        batch_idx: int,
    ) -> None:
        image = batch["image"]
        result = self(image)
        loss = torch.mean(torch.abs(result["restored"] - image))

        self.log(
            "val/loss",
            loss,
            batch_size=image.shape[0],
        )

    def predict_step(
        self,
        batch: PrivateImageBatch,
        batch_idx: int,
    ) -> ExternalAutoencoderPrediction:
        image = batch["image"]
        result = self(image)
        attributes = batch.get("attributes")

        return ExternalAutoencoderPrediction(
            embedding=result["features"],
            input=image,
            reconstruction=result["restored"],
            sample_id=batch.get("identifier"),
            label=(
                attributes.get("category")
                if attributes is not None
                else None
            ),
            metadata=attributes,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )


@dataclass(slots=True)
class MissingReconstructionPrediction:
    """Structurally invalid because reconstruction is required for AEs."""

    embedding: torch.Tensor
    input: torch.Tensor


class MissingReconstructionExternalAutoencoder(
    CompatibleExternalAutoencoder
):
    def predict_step(
        self,
        batch: AutoencoderBatch,
        batch_idx: int,
    ) -> MissingReconstructionPrediction:
        image = batch["x"]
        restored_image, representation = self(image)

        return MissingReconstructionPrediction(
            embedding=representation,
            input=image,
        )