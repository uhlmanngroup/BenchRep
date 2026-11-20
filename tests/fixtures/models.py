from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired
from typing_extensions import TypedDict

import lightning as L
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

        self.latent_dim = latent_dim
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
            "losses/train/total_weighted_loss",
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
            "losses/val/total_weighted_loss",
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
            "losses/train/total_weighted_loss",
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
            "losses/val/total_weighted_loss",
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


class TinyYOYOModel(L.LightningModule):
    """Small trainable model with private output names and deliberate export faults.

    Used by workflow integration tests; no canonical base class or prediction
    annotation is supplied. All computations use the existing synthetic data.
    """

    def __init__(
        self,
        initial_gain: float | None = 0.25,
        input_key: str = "x",
        output_kind: str = "mapping",
        fault: str | None = None,
        emit_metadata: bool = True,
    ) -> None:
        super().__init__()
        value = torch.rand(()) if initial_gain is None else torch.tensor(initial_gain)
        self.gain = nn.Parameter(value)
        self.input_key = input_key
        self.output_kind = output_kind
        self.fault = fault
        self.emit_metadata = emit_metadata
        self.initial_weight_seen: float | None = None

    def forward(self, image):
        return image * self.gain

    def training_step(self, batch, batch_idx):
        if self.initial_weight_seen is None:
            self.initial_weight_seen = float(self.gain.detach())
        image = batch[self.input_key]
        return (self(image) - image).square().mean()

    def validation_step(self, batch, batch_idx):
        image = batch[self.input_key]
        self.log(
            "val/custom", (self(image) - image).square().mean(), batch_size=len(image)
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def predict_step(self, batch, batch_idx):
        from types import SimpleNamespace

        image = batch[self.input_key]
        score = image.flatten(1).mean(1) * self.gain
        features = torch.stack((score, self.gain.expand_as(score)), dim=1)
        if self.output_kind == "tensor":
            return features
        result = {
            "features": features,
            "score": score,
            "column": score[:, None],
            "image": image,
            "decoded": self(image),
        }
        if self.emit_metadata:
            result["sample_id"] = batch.get("sample_id", batch.get("identifier"))
            result["label"] = batch.get("label")
            result["metadata"] = batch.get("metadata", batch.get("attributes"))
        if self.fault == "missing_primary":
            result.pop("features")
        elif self.fault == "scalar_primary":
            result["features"] = score
        elif self.fault == "image_vector":
            result["column"] = image
        elif self.fault == "batch_alignment":
            # Total row counts still match across the first two batches.
            result["score"] = (
                torch.cat((score, score[:1]))
                if batch_idx == 0
                else score[1:] if batch_idx == 1 else score
            )
        elif self.fault == "metadata_alignment":
            result["label"] = (
                list(batch["label"]) + [0]
                if batch_idx == 0
                else (
                    list(batch["label"])[1:] if batch_idx == 1 else list(batch["label"])
                )
            )
        elif self.fault == "metadata_keys" and batch_idx == 1:
            result["metadata"] = {"different": [1] * len(image)}
        elif self.fault == "reconstruction_alignment":
            result["decoded"] = (
                torch.cat((image, image[:1]))
                if batch_idx == 0
                else image[1:] if batch_idx == 1 else image
            )
        elif self.fault == "missing_input":
            result.pop("image")
        elif self.fault == "runtime":
            raise RuntimeError("deliberate YOYO predict_step failure")
        elif self.fault == "reserved_names":
            result = {"label": features, "sample_id": score, "metadata": score[:, None]}
        elif self.fault == "annotation_collision":
            result["metadata"] = {"score": score}
        return SimpleNamespace(**result) if self.output_kind == "attributes" else result
