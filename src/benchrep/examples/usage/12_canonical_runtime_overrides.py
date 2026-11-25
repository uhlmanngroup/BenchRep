"""Supply a custom autoencoder and datamodule to canonical workflows.

Run from your working directory:

    python -m benchrep.examples.usage.12_canonical_runtime_overrides

Training constructs both classes using parameters from the YAML. Prediction
reuses their instances and loads the best training checkpoint. The datamodule
returns ordinary (image, label) batches; the model still returns BenchRep's
canonical autoencoder prediction output.

The first run downloads Fashion-MNIST. Training runs for five epochs, followed by
embedding and reconstruction export on the test split and evaluation.
"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from benchrep import (
    EvaluationOverrides,
    PredictionOverrides,
    TrainingOverrides,
    evaluate,
    predict_ae,
    train_ae,
)
from benchrep.examples import example_config_path
from benchrep.interfaces.contracts import AutoencoderPredictionOutput
from benchrep.interfaces.models import BenchRepAutoencoderModel


EXAMPLE = "12_canonical_runtime_overrides"


class FashionMNISTAutoencoder(BenchRepAutoencoderModel):
    """A complete model implementation with the canonical prediction contract."""

    def __init__(self, latent_dim: int = 32, learning_rate: float = 0.001) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 784),
            nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))

    # Supplying both model and datamodule lets them agree on their own batch
    # format. A model-only override would need the canonical batch interface.
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, _ = batch
        loss = F.mse_loss(self(images), images)
        self.log("train/reconstruction_mse", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        images, _ = batch
        self.log("val/reconstruction_mse", F.mse_loss(self(images), images))

    def configure_optimizers(self):
        # A whole-model override owns its objective and optimizer implementation.
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Required by the family base, including this return annotation. These fields
    # let the standard prediction workflow export embeddings and reconstructions.
    def predict_step(self, batch, batch_idx: int) -> AutoencoderPredictionOutput:
        images, labels = batch
        embedding = self.encoder(images)
        return AutoencoderPredictionOutput(
            embedding=embedding,
            input=images,
            reconstruction=self.decoder(embedding),
            label=labels,
        )


class FashionMNISTDataModule(L.LightningDataModule):
    """Use Torchvision datasets directly, with no dataset registration."""

    def __init__(
        self,
        root: str = "data/fashion_mnist",
        batch_size: int = 128,
        val_fraction: float = 0.1,
        seed: int = 137,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.transform = v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        ])

    def prepare_data(self) -> None:
        FashionMNIST(self.root, train=True, download=True)
        FashionMNIST(self.root, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            dataset = FashionMNIST(self.root, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [1 - self.val_fraction, self.val_fraction],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage in (None, "predict"):
            self.predict_dataset = FashionMNIST(
                self.root, train=False, transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)


def main() -> None:
    training_result = train_ae(
        config=example_config_path(EXAMPLE, stage="training"),
        overrides=TrainingOverrides(
            model=FashionMNISTAutoencoder,
            datamodule=FashionMNISTDataModule,
        ),
    )

    # Reuse the model and datamodule instances returned by training, so prediction
    # needs no constructor arguments in its YAML. The selected checkpoint is still
    # loaded into the supplied model.
    prediction_result = predict_ae(
        config=example_config_path(EXAMPLE, stage="prediction"),
        overrides=PredictionOverrides(
            training_manifest_path=training_result.manifest_path,
            model=training_result.model,
            datamodule=training_result.datamodule,
        ),
    )
    evaluate(
        config=example_config_path(EXAMPLE, stage="evaluation"),
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_result.manifest_path,
        ),
    )


if __name__ == "__main__":
    main()
