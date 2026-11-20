"""Register a pretrained ConvNeXt encoder and train a canonical autoencoder.

Run from your working directory:

    python -m benchrep.examples.usage.11_pretrained_encoder_pipeline

Torchvision supplies ConvNeXt-Tiny and its ImageNet weights; no extra packages
are needed. The first run downloads the weights and STL-10. Training uses the
5,000-image labeled training split, with 10% held out for validation. Prediction
uses the test split and exports embeddings and reconstruction examples.

The encoder is fine-tuned alongside a newly initialized decoder for 10 epochs.
Images stay at STL-10's native 96 x 96 resolution. Lower trainer.max_epochs to
shorten the run, or datamodule.batch_size if GPU memory is insufficient.
"""

import gc

import torch
from torch import nn
from torchvision.models import convnext_tiny

from benchrep import (
    EvaluationOverrides,
    PredictionOverrides,
    evaluate,
    inspect_registry,
    predict_ae,
    train_ae,
)
from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentPort,
    ComponentTensorResult,
)
from benchrep.architecture.encoders import BaseEncoder
from benchrep.assembly.registries import ENCODERS
from benchrep.examples import example_config_path


EXAMPLE = "11_pretrained_encoder_pipeline"


class PretrainedConvNeXtEncoder(BaseEncoder):
    """ConvNeXt-Tiny features for RGB images with floating-point values in [0, 1].

    Parameters
    ----------
    input_shape
        Per-sample (channels, height, width). Requires three channels and
        spatial dimensions of at least 32 pixels.
    weights
        Torchvision weights identifier, or None for random initialization.
        ImageNet normalization is applied inside the encoder in either case.

    Notes
    -----
    Only the final classification layer is removed. Global average pooling
    and the pretrained output normalization remain, yielding 768 features.
    All remaining parameters are trainable.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        weights: str | None = "IMAGENET1K_V1",
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or input_shape[0] != 3:
            raise ValueError("input_shape must be (3, height, width).")
        if min(input_shape[1:]) < 32:
            raise ValueError("ConvNeXt requires height and width of at least 32.")

        self._input_shape = tuple(input_shape)
        self.backbone = convnext_tiny(weights=weights)
        self._output_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()

        # Normalize only the encoder input, keeping reconstruction targets in
        # [0, 1]. Buffers move with the model and are saved in its checkpoint.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    # Optional: describe the expected per-sample input shape.
    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    # Required by BaseEncoder: report the output vector size.
    @property
    def output_dim(self) -> int:
        return self._output_dim

    # Optional: lets BenchRep infer the decoder's initial feature-map shape.
    @property
    def feature_shape(self) -> tuple[int, int, int]:
        # ConvNeXt-Tiny reduces each spatial dimension by a factor of 32.
        # BenchRep uses this shape to initialize the upsampling decoder.
        return (
            self.output_dim, self.input_shape[1] // 32, self.input_shape[2] // 32,
        )

    # Required by BaseEncoder.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one feature vector per image, with shape (batch, 768)."""
        return self.backbone((x - self.mean) / self.std)


def main() -> None:
    # Register before config resolution, including in any new prediction process.
    # BaseEncoder provides canonical compatibility; the port contract also makes
    # this same component usable in a Composite model.
    ENCODERS.register(
        "pretrained_convnext",
        ArchitectureComponent(
            component=PretrainedConvNeXtEncoder,
            runtime_inputs=(
                ComponentPort(name="x", supported_structures=("image",)),
            ),
            runtime_result=ComponentTensorResult(supported_structures=("vector",)),
        ),
    )
    inspect_registry("encoder", "pretrained_convnext")

    training_result = train_ae(
        config=example_config_path(EXAMPLE, stage="training"),
    )
    training_manifest_path = training_result.manifest_path
    # Prediction constructs another model. Release the training objects first.
    del training_result
    gc.collect()

    prediction_result = predict_ae(
        config=example_config_path(EXAMPLE, stage="prediction"),
        overrides=PredictionOverrides(
            training_manifest_path=training_manifest_path,
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
