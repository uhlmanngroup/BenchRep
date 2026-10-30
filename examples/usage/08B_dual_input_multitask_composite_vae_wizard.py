"""Run a dual-input, multitask Composite VAE pipeline on enriched MNIST.

This example mirrors 08A_dual_input_multitask_composite_vae_yaml but instead
of using the YAMLs directly, we construct them using the workflow configuration
wizard GUI. However, for demonstration purposes, we will use the existing
08A YAMLs as a base that populates the configuration fields.

Imports and registers the custom dataset and two-input concatenation head from
example 08A.

Run from the repository root:

    python examples/usage/08B_dual_input_multitask_composite_vae_wizard.py
"""

from __future__ import annotations

from importlib import import_module

from benchrep import (
    configure_workflow,
    train_composite,
    predict_composite,
    evaluate,
    PredictionOverrides,
    EvaluationOverrides,
)
from benchrep.interfaces.model_families import COMPOSITE_FAMILY
from benchrep.architecture.composite_model_component_contracts import (
    ArchitectureComponent,
    ComponentPort,
    ComponentTensorResult,
)
from benchrep.assembly.registries import DATASETS, HEADS

example_08A = import_module("08A_dual_input_multitask_composite_vae_yaml")
CONFIG_DIR = example_08A.CONFIG_DIR
EnrichedMNISTDataset = example_08A.EnrichedMNISTDataset
ConcatenationHead = example_08A.ConcatenationHead


def main() -> None:
    # Register external components before launching configuration wizard.
    DATASETS.register(
        "enriched_mnist",
        EnrichedMNISTDataset,
    )

    HEADS.register(
        "concatenate",
        ArchitectureComponent(
            component=ConcatenationHead,
            runtime_inputs=(
                ComponentPort(
                    name="morphology",
                    supported_structures=("vector",),
                ),
                ComponentPort(
                    name="skeleton",
                    supported_structures=("vector",),
                ),
            ),
            runtime_result=ComponentTensorResult(
                supported_structures=("vector",),
            ),
        ),
    )

    print("\n=== Training ===")
    # Launch configuration wizard for the training workflow.
    # Optionally use the YAML from usage example 08A as a base.

    print(
        "\nTIP: For a quick demonstration, navigate to Trainer in the wizard "
        "and change 'Maximum epochs' to 1 before checking and saving the configuration.\n"
        "Prediction and evaluation wizards will open after their preceding "
        "workflows finish.\n"
    )

    training_wizard_result = configure_workflow(
        stage="training",
        config=CONFIG_DIR / "training.yaml",
        model_family=COMPOSITE_FAMILY,
        export_directory=CONFIG_DIR / "08B_wizard_configs",
    )

    # Stop this example if wizard was canceled.
    if training_wizard_result is None:
        print("Training configuration wizard cancelled. Aborting example execution.")
        return

    # Pass either the config object or the authored/resolved config path stored in the training_wizard_result
    # to the training workflow entrypoint.
    training_result = train_composite(
        config=training_wizard_result.config,
        # Optional and empty here; passing it preserves overrides if added to the launcher later.
        overrides=training_wizard_result.overrides,
    )
    print(f"Training manifest: {training_result.manifest_path}")

    print("\n=== Prediction ===")
    # Launch configuration wizard for the prediction workflow.
    # Optionally, use the YAML from usage example 08A as a base and pass the training manifest path in the overrides.
    prediction_wizard_result = configure_workflow(
        stage="prediction",
        config=CONFIG_DIR / "prediction.yaml",
        model_family=COMPOSITE_FAMILY,
        export_directory=CONFIG_DIR / "08B_wizard_configs",
        overrides=PredictionOverrides(
            training_manifest_path=training_result.manifest_path,
        ),
    )

    # Stop this example if wizard was canceled.
    if prediction_wizard_result is None:
        print("Prediction configuration wizard cancelled. Aborting example execution.")
        return

    # Recommend using the PredictionOverrides returned by the wizard instead of passing
    # the original or a new instance.
    prediction_result = predict_composite(
        config=prediction_wizard_result.config,
        overrides=prediction_wizard_result.overrides,
    )
    print(f"Prediction manifest: {prediction_result.manifest_path}")

    print("\n=== Evaluation ===")
    # Launch configuration wizard for the evaluation workflow.
    # Optionally, use the YAML from usage example 08A as a base and pass the prediction manifest path in the overrides.
    evaluation_wizard_result = configure_workflow(
        stage="evaluation",
        config=CONFIG_DIR / "evaluation.yaml",
        export_directory=CONFIG_DIR / "08B_wizard_configs",
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_result.manifest_path,
        ),
    )

    # Stop this example if wizard was canceled.
    if evaluation_wizard_result is None:
        print("Evaluation configuration wizard cancelled. Aborting example execution.")
        return

    evaluation_result = evaluate(
        config=evaluation_wizard_result.config,
        overrides=evaluation_wizard_result.overrides,
    )
    print(f"Evaluation manifest: {evaluation_result.manifest_path}")

    print("\n=== Pipeline complete ===")
    print(f"Training outputs:   {training_result.run_context.output_dir}")
    print(f"Prediction outputs: {prediction_result.run_context.output_dir}")
    print(f"Evaluation outputs: {evaluation_result.run_context.output_dir}")


if __name__ == "__main__":
    main()