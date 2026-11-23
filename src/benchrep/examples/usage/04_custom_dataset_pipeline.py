"""Run the canonical YAML-configured BenchRep pipeline with a custom dataset.

Run from your working directory:

    python -m benchrep.examples.usage.04_custom_dataset_pipeline

This example is similar to 01_yaml_pipeline but leverages the custom dataset
configuration feature to allow for the use of an external dataset not built
into BenchRep.
"""

from typing import Any, Literal

from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from benchrep import (
    inspect_registry,
    train_vae,
    predict_vae,
    evaluate,
    PredictionOverrides,
    EvaluationOverrides,
)
from benchrep.architecture.data import BaseDataset
from benchrep.assembly.registries import DATASETS
from benchrep.examples import example_config_path
from benchrep.records import print_divider


EXAMPLE = "04_custom_dataset_pipeline"


# Adapt an external dataset to the BenchRep dataset contract.
class FashionMNISTDataset(BaseDataset):
    """Fashion-MNIST adapter implementing BenchRep's dataset contract."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        download: bool = False,
    ) -> None:
        super().__init__()

        # CustomDatasetConfig.params is an untyped mapping, so validate values
        # whose incorrect interpretation could otherwise become a silent error.
        if split not in {"train", "test"}:
            raise ValueError(
                "split must be 'train' or 'test', "
                f"got {split!r}."
            )

        # This only adapts PIL images to tensors. Configurable preprocessing and
        # augmentation remain controlled by BenchRep's transform pipeline.
        self.dataset = FashionMNIST(
            root=root,
            train=split == "train",
            transform=v2.ToImage(),
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        x, y = self.dataset[index]

        # BenchRep datasets return a mapping rather than torchvision's
        # `(image, target)` tuple. `x` is the required model input, while
        # `label` and `sample_id` are optional fields used by prediction,
        # evaluation, stratified export, and provenance.
        sample = {
            "x": x,
            "label": y,
            "sample_id": index,
        }

        # Validate the adapted sample against the shared dataset contract
        # before returning it to BenchRep's datamodule and transform pipeline.
        return self.validate_sample(sample)


def main() -> None:
    # Registration must happen before workflow config resolution and must be
    # repeated in any future process reconstructing this run from resolved YAML.
    DATASETS.register(
        "fashion_mnist",
        FashionMNISTDataset,
        "fashionmnist",
    )

    # Confirm that the custom component is available through the public registry.
    inspect_registry("dataset", "fashion_mnist")

    print_divider("Pipeline", major=True)
    training_result = train_vae(
        config=example_config_path(EXAMPLE, stage="training"),
    )
    print(f"Training manifest: {training_result.manifest_path}")

    prediction_result = predict_vae(
        config=example_config_path(EXAMPLE, stage="prediction"),
        overrides=PredictionOverrides(
            training_manifest_path=training_result.manifest_path,
        ),
    )
    print(f"Prediction manifest: {prediction_result.manifest_path}")

    evaluation_result = evaluate(
        config=example_config_path(EXAMPLE, stage="evaluation"),
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_result.manifest_path,
        ),
    )
    print(f"Evaluation manifest: {evaluation_result.manifest_path}")

    print_divider("Pipeline", end=True, major=True)
    print(f"Training outputs:   {training_result.run_context.output_dir}")
    print(f"Prediction outputs: {prediction_result.run_context.output_dir}")
    print(f"Evaluation outputs: {evaluation_result.run_context.output_dir}")


if __name__ == "__main__":
    main()