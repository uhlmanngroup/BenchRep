"""Run the canonical YAML-configured BenchRep pipeline with a custom loss.

Run from the repository root:

    python examples/usage/05_custom_loss_pipeline.py

This example is similar to 01_yaml_pipeline but registers a custom image
gradient reconstruction loss and combines it with BenchRep's built-in MSE
reconstruction loss and Gaussian KL regularization.
"""

from pathlib import Path

import torch

from benchrep import (
    inspect_registry,
    train_vae,
    predict_vae,
    evaluate,
)
from benchrep.architecture.losses import BaseReconstructionLoss
from benchrep.assembly.registries import RECONSTRUCTION_LOSSES


# Resolve config paths relative to this script.
CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "05_custom_loss_pipeline"
)

# How custom losses move through BenchRep:
#
# 1. `gradient_difference` should be under `losses.reconstruction` in training.yaml.
# 2. The registration in `main()` connects that name to this Python class.
# 3. BenchRep instantiates the class using the configured `params`. This loss
#    has no constructor parameters, so its `params` mapping is empty in the YAML.
# 4. BenchRep wraps the instantiated loss and its configured `weight` in a
#    `LossTerm` dataclass, then stores it under the same name in a plain
#    dictionary passed to the VAE. The VAE receives a separate dictionary for
#    each loss role: reconstruction and regularization.
# 5. The VAE calls `forward(reconstruction, target)` during each loss step,
#    multiplies the returned raw loss by that weight, and adds it to the other
#    configured reconstruction and regularization losses.
#
# Users therefore implement and register the loss module, but do not need to
# instantiate it, package it with its weight, or call it from the model.
class GradientDifferenceLoss(BaseReconstructionLoss):
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


def main() -> None:
    # Connect the `gradient_difference` key used in training.yaml to the custom
    # loss class. Registration must happen before workflow config resolution and
    # must be repeated in any future process reconstructing this run.
    RECONSTRUCTION_LOSSES.register(
        "gradient_difference",
        GradientDifferenceLoss,
        "gdl",
    )

    # Confirm that the custom loss is available through the public registry.
    inspect_registry(
        "reconstruction_loss",
        "gradient_difference",
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