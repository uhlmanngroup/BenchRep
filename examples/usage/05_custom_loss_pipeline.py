"""Run a canonical VAE pipeline with two registered custom losses.

This example registers a reconstruction loss using the canonical
reconstruction interface and a custom objective using BenchRep's fixed
full-context interface. Neither registration declares Composite runtime inputs.

Both losses are combined with BenchRep's built-in MSE reconstruction loss and
Gaussian KL regularization.

Run from the repository root:

    python examples/usage/05_custom_loss_pipeline.py
"""

from pathlib import Path
from collections.abc import Mapping
from typing import Any

import torch

from benchrep import (
    inspect_registry,
    train_vae,
    predict_vae,
    evaluate,
)
from benchrep.architecture.losses import (
    BaseCustomObjectiveLoss,
    LossComponent,
)
from benchrep.assembly.registries import RECONSTRUCTION_LOSSES, CUSTOM_OBJECTIVE_LOSSES


# Resolve config paths relative to this script.
CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "05_custom_loss_pipeline"
)

# How losses work in BenchRep:
#
# Training config groups loss terms by role. Each term selects a registered
# loss, provides its constructor `params`, and assigns the scalar `weight`
# applied to its output.
#
# Canonical models supply fixed runtime inputs. Reconstruction losses receive
# `reconstruction` and `target`, while custom objectives receive the complete
# `batch` and `model_output` mappings. Because this example uses a canonical
# VAE, neither registered loss needs a Composite runtime-input contract.
#
# Custom objectives must subclass BaseCustomObjectiveLoss because they use
# BenchRep's fixed unrestricted interface.
class GradientDifferenceLoss(torch.nn.Module):
    """Penalize differences in horizontal and vertical image gradients."""

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if reconstruction.shape != target.shape:
            raise ValueError(
                "reconstruction and target must have the same shape, got "
                f"{tuple(reconstruction.shape)} and "
                f"{tuple(target.shape)}."
            )

        if reconstruction.ndim < 2:
            raise ValueError(
                "GradientDifferenceLoss requires at least two spatial "
                "dimensions."
            )

        if reconstruction.shape[-2] < 2 or reconstruction.shape[-1] < 2:
            raise ValueError(
                "Image height and width must both be at least 2."
            )

        # Measure the magnitude of every horizontal neighboring-pixel change.
        reconstruction_dx = (
            reconstruction[..., 1:] - reconstruction[..., :-1]
        ).abs()
        target_dx = (
            target[..., 1:] - target[..., :-1]
        ).abs()

        # Measure the magnitude of every vertical neighboring-pixel change.
        reconstruction_dy = (
            reconstruction[..., 1:, :] - reconstruction[..., :-1, :]
        ).abs()
        target_dy = (
            target[..., 1:, :] - target[..., :-1, :]
        ).abs()

        # Compare reconstruction and target gradient magnitudes, then average
        # over every sample, channel, and spatial position in the batch.
        horizontal_loss = (
            reconstruction_dx - target_dx
        ).abs().mean()

        vertical_loss = (
            reconstruction_dy - target_dy
        ).abs().mean()

        return horizontal_loss + vertical_loss


class AuxiliaryLatentClassificationLoss(BaseCustomObjectiveLoss):
    """Encourage VAE embeddings to predict the input class."""
    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        # Simple linear classifier
        self.classifier = torch.nn.Linear(
            embedding_dim,
            n_classes,
        )

    def forward(
        self,
        *,
        batch: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        labels = batch["label"]
        embeddings = model_output["embedding"]

        if not isinstance(labels, torch.Tensor):
            raise TypeError(
                "batch['label'] must be a torch.Tensor."
            )

        if labels.ndim != 1:
            raise ValueError(
                "batch['label'] must be one-dimensional."
            )

        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "Labels and embeddings must have the same batch size."
            )

        logits = self.classifier(embeddings)

        return torch.nn.functional.cross_entropy(logits, labels.long())


def main() -> None:
    # Register a reconstruction loss for canonical use. LossComponent carries
    # the loss class, while omitted runtime_inputs means that no Composite
    # contract is declared.
    RECONSTRUCTION_LOSSES.register(
        "gradient_difference",
        LossComponent(GradientDifferenceLoss),
        "gdl",
    )

    # Confirm that the custom loss is available through the public registry.
    inspect_registry(
        "reconstruction_loss",
        "gradient_difference",
    )

    # Register a custom objective. Its BaseCustomObjectiveLoss interface fixes
    # the runtime inputs, so no per-loss runtime contract is declared.
    CUSTOM_OBJECTIVE_LOSSES.register(
        "auxiliary_latent_classification",
        LossComponent(AuxiliaryLatentClassificationLoss),
        "latent_classification",
    )

    # Confirm that the custom loss is available through the public registry.
    inspect_registry(
        "custom_objective_loss",
        "auxiliary_latent_classification",
    )

    print("\n=== Training ===")
    training_result = train_vae(
        config_path=CONFIG_DIR / "training.yaml",
    )
    print(f"Training manifest: {training_result.manifest_path}")

    print("\n=== Prediction ===")
    prediction_result = predict_vae(
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