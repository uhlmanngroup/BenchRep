"""Train a YOYO classifier with BenchRep's configured data pipeline.

Requires BenchRep's scverse extra for t-SNE. Run from your working directory:

    python -m benchrep.examples.usage.13_yoyo_classifier_pipeline

The supplied LightningModule owns its classification loss, optimizer, and
prediction output. It has no decoder or reconstruction output, so it does not
implement the canonical AE/VAE contracts. YOYO lets us use this complete Python
implementation directly, without describing it as a Composite assembly.

The first run downloads CIFAR-10. Training runs for ten epochs; prediction
exports features and logits on the test split. Evaluation measures the features
through clustering and classification probes and exports PCA/t-SNE figures.
Lower trainer.max_epochs or add batch limits for a shorter run.
"""

import gc

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from benchrep import (
    EvaluationOverrides,
    PredictionOverrides,
    TrainingOverrides,
    evaluate,
    predict_yoyo,
    train_yoyo,
)
from benchrep.examples import example_config_path


EXAMPLE = "13_yoyo_classifier_pipeline"


class CIFAR10Classifier(L.LightningModule):
    """A classifier whose pooled features are also useful for export."""

    def __init__(
        self,
        feature_dim: int = 128,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        return {"features": features, "logits": self.classifier(features)}

    def _step(self, batch: dict[str, torch.Tensor], split: str) -> torch.Tensor:
        output = self(batch["x"])
        labels = batch["label"].long()
        loss = F.cross_entropy(output["logits"], labels)
        accuracy = (output["logits"].argmax(dim=1) == labels).float().mean()
        self.log(f"{split}/cross_entropy", loss, on_step=False, on_epoch=True)
        self.log(f"{split}/accuracy", accuracy, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        # YOYO uses explicit export keys instead of declared output roles.
        # label and sample_id are recognized prediction metadata.
        return {
            **self(batch["x"]),
            "label": batch["label"],
            "sample_id": batch["sample_id"],
        }


def main() -> None:
    # Only the model is external. BenchRep builds the registered dataset,
    # transform pipelines, subsets, and dataloaders from the training YAML.
    training_result = train_yoyo(
        config=example_config_path(EXAMPLE, stage="training"),
        overrides=TrainingOverrides(model=CIFAR10Classifier),
    )
    training_manifest_path = training_result.manifest_path
    del training_result
    gc.collect()

    # Supply the class again; constructor parameters are inherited from training.
    # Prediction constructs it and loads the best checkpoint.
    prediction_result = predict_yoyo(
        config=example_config_path(EXAMPLE, stage="prediction"),
        overrides=PredictionOverrides(
            training_manifest_path=training_manifest_path,
            model=CIFAR10Classifier,
        ),
    )
    prediction_manifest_path = prediction_result.manifest_path
    del prediction_result
    gc.collect()

    evaluate(
        config=example_config_path(EXAMPLE, stage="evaluation"),
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_manifest_path,
        ),
    )


if __name__ == "__main__":
    main()
