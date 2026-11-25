"""Export and evaluate pretrained ViT features without a BenchRep training run.

Requires BenchRep's scverse extra for t-SNE. Run with:

    python -m benchrep.examples.usage.14_pretrained_yoyo_prediction

The first run downloads Torchvision's ImageNet-pretrained ViT-B/16 weights and
STL-10. A custom datamodule applies the weights' preprocessing transforms to
STL-10's 5,000 labeled training images. The classifier is removed to export
768-dimensional image embeddings.

The supplied model instance already contains its weights; no training manifest
or BenchRep checkpoint is needed. Evaluation runs cross-validated probes on the
frozen embeddings and exports clustering metrics and PCA/t-SNE figures.
Standalone prediction writes its run under outputs/ in the working directory.

Lower BATCH_SIZE for less GPU memory use, or set data.max_batches in the
prediction YAML to process fewer images. STL-10's archive includes its other
splits, even though this example reads only the labeled training split.
"""

import gc

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.models import ViT_B_16_Weights, vit_b_16

from benchrep import EvaluationOverrides, PredictionOverrides, evaluate, predict_yoyo
from benchrep.examples import example_config_path


EXAMPLE = "14_pretrained_yoyo_prediction"
BATCH_SIZE = 32


class PretrainedViT(L.LightningModule):
    """Expose a pretrained backbone's global image features for prediction."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        images, labels = batch
        return {
            "embedding": self(images),
            "label": labels,
        }


class STL10DataModule(L.LightningDataModule):
    """Apply the pretrained weights' transforms and provide prediction batches."""

    def __init__(
        self, transform, root: str = "data/stl10", batch_size: int = BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.root = root
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        STL10(self.root, split="train", download=True)

    def setup(self, stage: str | None = None) -> None:
        self.dataset = STL10(self.root, split="train", transform=self.transform)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
        )


def main() -> None:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    backbone = vit_b_16(weights=weights)
    # Return the CLS-token embedding before the ImageNet classification head.
    backbone.heads = nn.Identity()
    model = PretrainedViT(backbone)
    datamodule = STL10DataModule(weights.transforms(), batch_size=BATCH_SIZE)

    # Supply initialized instances; BenchRep runs prediction with their current
    # weights and preprocessing, without a training manifest or checkpoint.
    prediction_result = predict_yoyo(
        config=example_config_path(EXAMPLE, stage="prediction"),
        overrides=PredictionOverrides(model=model, datamodule=datamodule),
    )
    prediction_manifest_path = prediction_result.manifest_path
    del prediction_result, model, backbone, datamodule
    gc.collect()

    evaluate(
        config=example_config_path(EXAMPLE, stage="evaluation"),
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_manifest_path,
        ),
    )


if __name__ == "__main__":
    main()
