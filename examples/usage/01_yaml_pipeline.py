"""Run the canonical YAML-configured BenchRep pipeline.

Run from the repository root:

    python examples/usage/01_yaml_pipeline.py

Each workflow writes its own outputs, manifest, runtime-environment record,
and audit report. The generated manifests connect the three stages.
"""

from pathlib import Path

from benchrep import evaluate, predict_vae, train_vae


# Resolve config paths relative to this script
CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "01_yaml_pipeline"
)


def main() -> None:
    print("\n=== Training ===")
    training_result = train_vae(
        config_path=CONFIG_DIR / "training.yaml",
    )
    print(f"Training manifest: {training_result.manifest_path}")

    print("\n=== Prediction ===")
    prediction_result = predict_vae(
        config_path=CONFIG_DIR / "prediction.yaml",
        # Use the manifest generated above instead of hard-coding an output path
        # in prediction.yaml.
        training_manifest_path=training_result.manifest_path,
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