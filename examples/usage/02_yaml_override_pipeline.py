"""Run a BenchRep pipeline using typed overrides on existing YAML configs.

Run from the repository root:

    python examples/usage/02_yaml_override_pipeline.py

This example adapts the MNIST VAE pipeline from 01_yaml_pipeline.py to
CIFAR-10. It demonstrates registry and schema inspection, top-level config
replacement, and export of a harmonized resolved YAML.
"""

from pathlib import Path

from benchrep import (
    inspect_registry,
    inspect_config,
    train_vae,
    predict_vae,
    evaluate,
)
from benchrep.assembly.schemas import (
    # Training schema
    EncoderConfig,
    DecoderConfig,
    RunConfig,
    CIFAR10DatasetConfig,
    CIFAR10DatasetParams,
    TrainingConfig,
)

# Resolve config paths relative to this script.
CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "01_yaml_pipeline"
)


def main() -> None:
    # Inspect the available component registries.
    inspect_registry()

    # Inspect the "dataset" registry.
    inspect_registry("dataset")

    # Inspect the registered CIFAR-10 Dataset class more closely
    inspect_registry("dataset", "cifar_10")

    # Inspect the top level `TrainingConfig` to understand which configuration needs to be overridden
    inspect_config(TrainingConfig)

    # Dataset configuration is controlled by TrainingConfig.dataset, with CIFAR10DatasetConfig as an accepted type
    # Let's inspect the expected structure for CIFAR10DatasetConfig
    inspect_config(CIFAR10DatasetConfig)
    # Let's inspect the nested `CIFAR10DatasetParams`
    inspect_config(CIFAR10DatasetParams)
    # Inspect the registered architecture components that must be adapted to
    # CIFAR-10's three-channel 32 x 32 input shape.
    inspect_registry("encoder", "conv2d")
    inspect_registry("decoder", "upsample_conv2d")

    # First, let's customize the project name by overriding the run
    run_config = RunConfig(
        # A config component replaces the complete matching YAML section.
        # Omitted fields therefore use schema defaults rather than original YAML values.
        output_root=Path("outputs"),
        project_name="02_yaml_override_pipeline",
    )

    # Build separate dataset configs for the training and test splits.
    cifar_10_training_config = CIFAR10DatasetConfig(
        name="cifar10",
        params=CIFAR10DatasetParams(
            root=Path("examples/data/cifar10"),
            split="train",
            download=True,
        ),
    )

    cifar_10_prediction_config = CIFAR10DatasetConfig(
        name="cifar10",
        params=CIFAR10DatasetParams(
            root=Path("examples/data/cifar10"),
            split="test",
            download=True,
        ),
    )

    # Since CIFAR-10 has a different input shape, let's also update the encoder and decoder parameters.
    encoder_config = EncoderConfig(
        name="conv2d",
        params={
            "input_shape": [3, 32, 32],
            "output_dim": 32,
            "channels": [32, 64],
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "normalization": "batchnorm",
        },
    )
    # The decoder output shape must also match the new dataset shape.
    decoder_config = DecoderConfig(
        name="upsample_conv2d",
        params={
            "output_shape": [3, 32, 32],
            "channels": [32, 16],
            "normalization": "batchnorm",
            "output_activation": "sigmoid",
        },
    )

    # Configuration components prepared, we can now begin the pipeline
    print("\n=== Training ===")
    training_result = train_vae(
        config_path=CONFIG_DIR / "training.yaml",
        # Mapping keys match the top-level field names in `TrainingConfig`.
        config_components={
            "run": run_config,
            "encoder": encoder_config,
            "decoder": decoder_config,
            "dataset": cifar_10_training_config,
        },
    )
    # BenchRep writes the harmonized effective configuration as resolved_config.yaml.
    # It can be supplied directly as config_path in a future run when the manifest
    # records `provenance.config.run_reconstructable_from_resolved_config: true`.
    resolved_training_config_path = (
        training_result.run_context.config_dir
        / "resolved_config.yaml"
    )
    print(f"Training manifest: {training_result.manifest_path}")
    print(f"Resolved training config: {resolved_training_config_path}")

    print("\n=== Prediction ===")
    prediction_result = predict_vae(
        config_path=CONFIG_DIR / "prediction.yaml",
        # Use the manifest generated above instead of hard-coding an output path
        # in prediction.yaml.
        training_manifest_path=training_result.manifest_path,
        # prediction.yaml selects the MNIST test split, so replace its complete
        # dataset section with the corresponding CIFAR-10 test configuration.
        config_components={
            "dataset": cifar_10_prediction_config,
        },
    )
    print(f"Prediction manifest: {prediction_result.manifest_path}")

    print("\n=== Evaluation ===")
    evaluation_result = evaluate(
        config_path=CONFIG_DIR / "evaluation.yaml",
        # The prediction manifest supplies the exported embeddings,
        # reconstructions, provenance, and inherited run identity.
        prediction_manifest_path=prediction_result.manifest_path,
    )
    print(f"Evaluation manifest: {evaluation_result.manifest_path}")

    print("\n=== Pipeline complete ===")
    print(f"Training outputs:   {training_result.run_context.output_dir}")
    print(f"Prediction outputs: {prediction_result.run_context.output_dir}")
    print(f"Evaluation outputs: {evaluation_result.run_context.output_dir}")


if __name__ == "__main__":
    main()