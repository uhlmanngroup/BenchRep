"""Run the canonical YAML-configured BenchRep pipeline.

Run from your working directory:

    python -m benchrep.examples.usage.01_yaml_pipeline

Each workflow writes its own outputs, manifest, and runtime-environment record.
The generated manifests connect the three stages.
"""


from benchrep import (
    train_vae,
    predict_vae,
    evaluate,
    PredictionOverrides,
    EvaluationOverrides,
)
from benchrep.examples import example_config_path
from benchrep.records import print_divider



EXAMPLE = "01_yaml_pipeline"


def main() -> None:
    print_divider("Pipeline", major=True)

    training_result = train_vae(
        config=example_config_path(EXAMPLE, stage="training"),
    )
    print(f"Training manifest: {training_result.manifest_path}")

    prediction_result = predict_vae(
        config=example_config_path(EXAMPLE, stage="prediction"),
        # Use the manifest generated above instead of hard-coding an output path
        # in prediction.yaml.
        overrides=PredictionOverrides(
            training_manifest_path=training_result.manifest_path
        ),
    )
    print(f"Prediction manifest: {prediction_result.manifest_path}")

    evaluation_result = evaluate(
        config=example_config_path(EXAMPLE, stage="evaluation"),
        # The prediction manifest supplies the exported AnnData,
        # reconstructions, provenance, and inherited run identity.
        overrides=EvaluationOverrides(
            prediction_manifest_path=prediction_result.manifest_path
        ),
    )
    print(f"Evaluation manifest: {evaluation_result.manifest_path}")

    print_divider("Pipeline", end=True, major=True)
    print(f"Training outputs:   {training_result.run_context.output_dir}")
    print(f"Prediction outputs: {prediction_result.run_context.output_dir}")
    print(f"Evaluation outputs: {evaluation_result.run_context.output_dir}")


if __name__ == "__main__":
    main()